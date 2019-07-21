#include <string>
#include <json/json.h>
#include <iostream>
#include <fstream>

using std::ofstream;
using std::ifstream;
using std::getline;

int ReadJson(const std::string& strValue);
std::string writeJson();

int main(int argc, char** argv)
{
  using namespace std;

  std::string ofile = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/"
                      "retinanet_ratio32_bilinear/example.txt";

//  ofstream out(ofile);
//  cout << "--------------------------------" << endl;
//  string strMsg = writeJson();
//  cout<< "json write : " << endl << strMsg << endl;
//  out << strMsg;
//  out.close();

  string one_line;
  string all_lines;
  ifstream ifile(ofile);
  if (ifile.is_open()) {
    while(!ifile.eof()) {
      getline(ifile, one_line);
      all_lines += one_line + "\n";
    }
  }
  cout << "string from file:" << endl;
  cout << all_lines << endl;

  cout << "--------------------------------" << endl;
  cout << "json read:" << endl;
  ReadJson(all_lines);
  cout << "--------------------------------" << endl;

  return 0;
}

int ReadJson(const std::string& strValue)
{
  using namespace std;

  Json::Reader reader;
//  Json::CharReader reader;
//  Json::CharReaderBuilder builder;
//  builder["collectComments"] = false;
//  JSONCPP_STRING errs;
  Json::Value value;
//  bool ok = parseFromStream(builder, strValue, &value, &errs);

  if (reader.parse(strValue, value))
  {
    string out = value["name"].asString();
    cout << "name : "   << out << endl;
    cout << "number : " << value["number"].asInt() << endl;
    cout << "value : "  << value["value"].asBool() << endl;
    cout << "no such num : " << value["haha"].asInt() << endl;
    cout << "no such str : " << value["hehe"].asString() << endl;

    const Json::Value arrayNum = value["arrnum"];
    for (unsigned int i = 0; i < arrayNum.size(); i++)
    {
      cout << "arrnum[" << i << "] = " << arrayNum[i] << endl;
    }

    Json::Value array2 = value["array2"];
    for (unsigned int i = 0; i < array2.size(); i++)
    {
      cout << "array2[" << i << "] = " << array2[i].asFloat() << endl;
    }

    cout << "bool type bool_a: " << value["bool_a"].asBool() << endl;

    Json::Value arrayObj = value["array"];
    cout << "array size = " << arrayObj.size() << endl;
    for(unsigned int i = 0; i < arrayObj.size(); i++)
    {
      cout << arrayObj[i];
    }
    for(unsigned int i = 0; i < arrayObj.size(); i++)
    {
      if (arrayObj[i].isMember("string"))
      {
        out = arrayObj[i]["string"].asString();
        std::cout << "string : " << out << std::endl;
      }
    }
  }

  return 0;
}

std::string writeJson()
{
  using namespace std;

  Json::Value root;
  Json::Value arrayObj;
  Json::Value item;
  Json::Value iNum;

  item["string"]    = "this is a string";
  item["number"]    = 999;
  item["aaaaaa"]    = "bbbbbb";
  arrayObj.append(item);

  //直接对jsoncpp对象以数字索引作为下标进行赋值，则自动作为数组
  iNum[1] = 1;
  iNum[2] = 2;
  iNum[3] = 3;
  iNum[4] = 4;
  iNum[5] = 5;
  iNum[6] = 6;

  //增加对象数组
  root["array"]    = arrayObj;
  //增加字符串
  root["name"]    = "json";
  //增加数字
  root["number"]    = 666;
  //增加布尔变量
  root["value"]    = true;
  //增加数字数组
  root["arrnum"]    = iNum;

  root.toStyledString();
  string out = root.toStyledString();

  return out;
}

