//
// Created by DefTruth on 2021/6/19.
//

#ifndef LITE_AI_ORT_CORE_ORT_DEFS_H
#define LITE_AI_ORT_CORE_ORT_DEFS_H

#include "lite/lite.ai.defs.h"

#define LITEORT_DEBUG 1

//#ifndef LITEHUB_EXPORTS
//# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
//#   define LITEHUB_EXPORTS __declspec(dllexport)
//# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
//#   define LITEHUB_EXPORTS __attribute__ ((visibility ("default")))
//# endif
//#endif
//
//#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
//#  define LITEHUB_WIN32
//#elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
//# define LITEHUB_UNIX
//#endif
//
//#ifndef LITEHUB_EXPORTS
//# define LITEHUB_EXPORTS
//#endif

#ifdef LITE_WIN32
# define LITEORT_CHAR wchar_t
#else
# define LITEORT_CHAR char
#endif

#ifdef LITE_WIN32
# define NONMINMAX
#endif

#endif //LITE_AI_ORT_CORE_ORT_DEFS_H
