# rmlk

Rusty Machine Learning Kit is a machine learning library.


Onnx version 1.13.1

## Todo

Implement


"Expand", 
"Sin",                  // NEEDS KERNEL, BACKEND AND TESTS
"Sigmoid",              // NEEDS KERNEL, BACKEND AND TESTS
"Cast",                 // NEEDS BACKEND & TESTS
"Unsqueeze",
"Shape",                // DONE
"ReduceMean",           // DONE
"Pow",                  // NEEDS KERNEL, BACKEND AND TESTS
"Gather",               // DONE
"Mul",                  // DONE
"Trilu", 
"Where",                // DONE
"Sqrt",                 // NEEDS BACKEND & TESTS
"Add",                  // DONE
"Neg",                  // NEEDS KERNEL, BACKEND AND TESTS
"Slice", 
"MatMul",
"ScatterND",            // NEEDS BACKEND AND MORE TESTS
"Equal",                // NEEDS KERNEL, BACKEND AND TESTS
"ConstantOfShape",      // DONE
"Transpose",            // DONE
"Range",
"Concat", 
"Div",                  // NEEDS KERNEL, BACKEND AND TESTS
"Greater",              // NEEDS KERNEL, BACKEND AND TESTS
"Softmax", 
"Reshape",              // DONE
"Cos"                   // NEEDS KERNEL, BACKEND AND TESTS

"Sub",                  // NEEDS KERNEL, BACKEND AND TESTS  
"Constant"
 

## Notes

* Our changes to definition of ConstantOfShape in terms of attributes needs to be considered in the converter.
* ReduceMean axes input should be in usize type.
* Check how we handle inputs of scattternd.