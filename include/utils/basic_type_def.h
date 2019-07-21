#ifndef C_OBJECT_FLOW_BASIC_TYPE_DEF_H
#define C_OBJECT_FLOW_BASIC_TYPE_DEF_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr.hpp>
#include <glog/logging.h>

#include "json/json.h"
#include "cnpy.h"

namespace cobjectflow {

typedef bool bool_t;
typedef signed char int8_t;
typedef short int int16_t;
typedef int int32_t;
typedef long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef float float32_t;
typedef double float64_t;

typedef Json::Value JsonValue;

struct Shape2D {
  int32_t h;
  int32_t w;
};

struct Shape3D {
  int32_t c;
  int32_t h;
  int32_t w;
};

struct Shape4D {
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
};

template<typename Dtype>
struct ImgMeta {
  /// original image shape, not used
  Shape3D ori_shape;
  /// rescaled image shape
  Shape3D img_shape;
  /// first rescaled, then paded image shape
  Shape3D pad_shape;
  /// map the predicted boxes to original image scale
  Dtype scale_factor;
  /// flip image flag, not used
  bool_t flip;
};

template<typename Dtype>
struct ImgInfo {
  float *img;
  std::vector<int32_t> img_shape;
  std::vector<int32_t> pad_shape;
  std::vector<Dtype> scale_factor;
};

template<typename Dtype>
struct Data2D {
  Dtype *data;
  Shape2D shape;
};

template<typename Dtype>
struct Data3D {
  Dtype *data;
  Shape3D shape;
};

template<typename Dtype>
struct Data4D {
  Dtype *data;
  Shape4D shape;
};

#define EXTRACT_JSON_VALUE_ARRAY_STRING(cfg, var_name) \
std::vector<std::string> var_name; \
JsonValue js_##var_name = cfg[#var_name]; \
for (int32_t i = 0; i < js_##var_name.size(); ++i) { \
var_name.push_back(js_##var_name[i].asString()); \
}

#define EXTRACT_JSON_VALUE_ARRAY_STRING_2D(cfg, var_name) \
std::vector<std::vector<std::string> > var_name; \
JsonValue js_##var_name = cfg[#var_name]; \
for (int32_t i = 0; i < js_##var_name.size(); ++i) { \
std::vector<std::string> var_temp; \
var_temp.push_back(js_##var_name[i][0].asString()); \
var_temp.push_back(js_##var_name[i][1].asString()); \
var_name.push_back(var_temp); \
}

#define EXTRACT_JSON_VALUE_ARRAY_INT32(cfg, var_name) \
std::vector<int32_t> var_name; \
JsonValue js_##var_name = cfg[#var_name]; \
for (int32_t i = 0; i < js_##var_name.size(); ++i) { \
var_name.push_back(js_##var_name[i].asInt()); \
}

#define EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(cfg, var_name, type) \
std::vector<Dtype> var_name; \
JsonValue js_##var_name = cfg[#var_name]; \
for (int32_t i = 0; i < js_##var_name.size(); ++i) { \
var_name.push_back(Dtype(js_##var_name[i].as##type())); \
}

//using boost::shared_array;
//#define boo_shared_ptr boost::shared_ptr
//#define std_shared_ptr std::shared_ptr

//inline void _assert_of(const char *msg, const char *reminder, const char *file,
//                       unsigned line) {
//  printf("assertion fail: %s\n%s%d\n%s\n", msg, file, line, reminder);
//  exit(0);
//}

//#define assert_of(expr, reminder) ((expr) || (_assert_of(#expr, (reminder), \
//                                  __FILE__, __LINE__), 0))

//template<typename T>
//std::shared_ptr<T> make_shared_array(size_t size)
//{
//  return std::shared_ptr<T>(new T[size], std::default_delete<T []>());
//}

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_BASIC_TYPE_DEF_H
