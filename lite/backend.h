//
// Created by DefTruth on 2021/5/30.
//

#ifndef LITEHUB_BACKEND_H
#define LITEHUB_BACKEND_H

#define BACKEND_ONNXRUNTIME
//#define BACKEND_NCNN
//#define BACKEND_MNN
//#define BACKEND_TNN

// ONNXRuntime
#ifdef BACKEND_ONNXRUNTIME

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// NCNN
#ifdef BACKEND_NCNN

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// MNN
#ifdef BACKEND_MNN

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// TNN
#ifdef BACKEND_TNN

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

#endif

#endif //LITEHUB_BACKEND_H
