use rmlk_runtime::Builder;
use rmlk_runtime::Value;
use std::io::BufRead;
use std::{collections::HashMap, fs};
use tokenizers::Tokenizer;

const NUM_LAYERS: usize = 28; // Transformer blocks.
const HEADS: usize = 8; // Attention heads per block.
const HEAD_DIM: usize = 128; // Hidden size of each head.
const VOCAB_SIZE: usize = 128256;

const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
const START_HEADER_ID: &str = "<|start_header_id|>";
const END_HEADER_ID: &str = "<|end_header_id|>";
const EOT_ID: &str = "<|eot_id|>";

fn kv_placeholder_shape(past_len: usize) -> (Vec<f32>, Vec<usize>) {
    let shape = vec![1, HEADS, past_len, HEAD_DIM];
    (Vec::new(), shape)
}

fn main() {
    env_logger::init();
    let mut args = std::env::args();
    args.next();

    let tokenizer_path = args.next().expect("missing tokenizer path argument");
    let model_path = args.next().expect("missing onnx model path argument");

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let graph = fs::read(model_path).expect("unable to read onnx model");

    let builder = Builder::with_model_from_memory(graph.into_boxed_slice()).unwrap();
    let mut engine = builder.build().unwrap();

    eprintln!("Runtime ready – type a prompt and hit <Enter>.\n");

    let mut input_ids: Vec<i64> = Vec::new();

    for line in std::io::stdin().lock().lines() {
        let Ok(user_input) = line else { continue };

        let full_prompt = format!(
            "{BEGIN_OF_TEXT}{START_HEADER_ID}system{END_HEADER_ID}\n\
            You are a helpful assistant.{EOT_ID}\n\
            {START_HEADER_ID}user{END_HEADER_ID}\n\
            {user_input}{EOT_ID}\n\
            {START_HEADER_ID}assistant{END_HEADER_ID}"
        );

        let enc = tokenizer.encode(full_prompt.as_str(), false).unwrap();
        input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));

        println!(
            "prompt (decoded): {:?}",
            tokenizer.decode(
                &input_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                true
            )
        );

        let mut output: Option<HashMap<String, Value>> = None;
        let mut past_sequence_len = 0usize;
        let mut input_tokens = input_ids.clone();
        loop {
            let seq_len = input_tokens.len();

            let input_tokens_shape = vec![1, seq_len];

            let position_ids: Vec<i64> =
                (past_sequence_len as i64..(past_sequence_len + seq_len) as i64).collect();

            let attention_mask: Vec<i64> = vec![1; past_sequence_len + seq_len];
            let attention_mask_shape = vec![1, past_sequence_len + seq_len];

            let mut feed: HashMap<String, Value> = HashMap::new();

            assert_eq!(input_tokens.len(), position_ids.len());
            assert_eq!(past_sequence_len + input_tokens.len(), attention_mask.len());

            past_sequence_len += input_tokens.len();

            feed.insert(
                "input_ids".into(),
                (input_tokens, input_tokens_shape.clone())
                    .try_into()
                    .unwrap(),
            );
            feed.insert(
                "attention_mask".into(),
                (attention_mask.clone(), attention_mask_shape.clone())
                    .try_into()
                    .unwrap(),
            );
            feed.insert(
                "position_ids".into(),
                (position_ids.clone(), input_tokens_shape.clone())
                    .try_into()
                    .unwrap(),
            );

            if let Some(out) = output.as_mut() {
                for layer in 0..NUM_LAYERS {
                    let layer_key_name = format!("present.{}.key", layer);
                    let layer_value_name = format!("present.{}.value", layer);

                    let (key, out_key_shape): (Vec<f32>, Vec<usize>) =
                        out.remove(&layer_key_name).unwrap().try_into().unwrap();
                    assert_eq!(key.len(), out_key_shape.iter().product::<usize>());

                    let (val, out_val_shape): (Vec<f32>, Vec<usize>) =
                        out.remove(&layer_value_name).unwrap().try_into().unwrap();
                    assert_eq!(val.len(), out_val_shape.iter().product::<usize>());

                    feed.insert(
                        format!("past_key_values.{}.key", layer),
                        (key, out_key_shape).try_into().unwrap(),
                    );
                    feed.insert(
                        format!("past_key_values.{}.value", layer),
                        (val, out_val_shape).try_into().unwrap(),
                    );
                }
            } else {
                for layer in 0..NUM_LAYERS {
                    let (kv_data, kv_shape) = kv_placeholder_shape(0);
                    assert!(kv_data.is_empty());

                    feed.insert(
                        format!("past_key_values.{}.key", layer),
                        (kv_data.clone(), kv_shape.clone()).try_into().unwrap(),
                    );
                    feed.insert(
                        format!("past_key_values.{}.value", layer),
                        (kv_data.clone(), kv_shape.clone()).try_into().unwrap(),
                    );
                }
            }

            output = Some(engine.run(feed).unwrap());

            let out = output.as_mut().expect("missing output");
            let logits: Vec<f32> = out.remove("logits").unwrap().try_into().unwrap();

            assert_eq!(logits.len(), seq_len * VOCAB_SIZE);

            let last_row = &logits[(seq_len - 1) * VOCAB_SIZE..seq_len * VOCAB_SIZE];

            let (next_id, _) = last_row
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            if next_id == 128001 || next_id == 128009 {
                println!();
                break;
            }

            let text = tokenizer.decode(&[next_id as u32], true).unwrap();

            print!("{text}");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            input_tokens = vec![next_id as i64];
        }

        println!();
        println!("Type a prompt and hit <Enter>.");
        input_ids.clear();
    }
}
