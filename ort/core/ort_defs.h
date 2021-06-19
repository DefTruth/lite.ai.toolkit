//
// Created by DefTruth on 2021/6/19.
//

#ifndef LITEHUB_ORT_CORE___ORT_CONFIG_H
#define LITEHUB_ORT_CORE___ORT_CONFIG_H
#include <iostream>

#define LITEORT_DEBUG 1

#ifndef LITEHUB_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define LITEHUB_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define LITEHUB_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define LITEHUB_WIN32
#elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
# define LITEHUB_UNIX
#endif

#ifndef LITEHUB_EXPORTS
# define LITEHUB_EXPORTS
#endif

#ifdef LITEHUB_WIN32
# define LITEHUBCHAR wchar_t
#else
# define LITEHUBCHAR char
#endif

#ifdef LITEHUB_WIN32
# define NONMINMAX
#endif

#ifndef __unused
# define __unused
#endif

#endif //LITEHUB_ORT_CORE___ORT_CONFIG_H
