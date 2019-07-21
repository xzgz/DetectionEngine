#include <iostream>
#include <fstream>
#include <pthread.h>

using namespace std;

int main() {
  ofstream time_record_file("./inference_time.txt");
  time_record_file << "asss";
}