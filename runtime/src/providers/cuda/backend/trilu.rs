/*
    for batch in 0..input.shape[0]:
       batch_offset = batch * input.strides[0]
       for row in 0..input.shape[1]:
           row_offset = batch_offset +  row * input.strides[1]

            if upper:
               start = max(k + row, 0)
               end = input.shape[2]
            else:
               start = 0
               end = min(k + row + 1, input.shape[2])

           for col in start..end:
               offset = row_offset +  col
               output.data[offset] = input.data[offset]

   [ 1   2  3  4
     5   6  7  8
     9  10 11 12
     13 14 15 16]

     (1, 4, 4)
     (16, 4, 1)

     // Given thread_idx

   offset = 0
   tmp_i = thread_idx
   for dim in 0..input.rank.rev():
       i_dim = tmp_i % input.shape[dim]
       offset += i_dim * input.strides[dim]
       tmp_i /= input.shape[dim]

   offset = 0
   tmp_i = thread_idx

   col = (tmp_i % input.shape[2])
   offset += col * input.strides[2]
   tmp_i /= input.shape[2]

   row = (tmp_i % input.shape[1])
   offset += row * input.strides[1]
   tmp_i /= input.shape[1]

   batch = (tmp_i % input.shape[0])
   offset += batch * input.strides[0]
   tmp_i /= input.shape[0]

    if upper:
       start = max(k + row, 0)
       end = input.shape[2]
    else:
       start = 0
       end = min(k + row + 1, input.shape[2])

    if col >= start && col < end:
       output[offset] = input[offset]
*/
