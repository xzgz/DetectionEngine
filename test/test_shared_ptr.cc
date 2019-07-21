//#include <iostream>
//#include <memory>
//
//#include "utils/basic_type_def.h"
//
//using namespace std;
//
//class Test
//{
// public:
//  Test(int a, string b = "") : a_(a), b_(b) {
//    c_ = new int[3];
//    c_[0] = 1;
//
////    shared_ptr<int> d_temp(new int[5]);
////    d_ = make_shared<int>(*(new int[3]));
////    d_ = make_shared<int>(*(new int(3)));
////    shared_ptr<int> d_temp(new int[5], [](int *p) {delete [] p;});
////    shared_ptr<int> d_temp(new int[5], default_delete<int []>());
////    d_ = d_temp;
//    d_ = make_shared_array<int>(5);
//    *d_ = 2;
//    cout << "in d_ uc: " << d_.use_count() << endl;
//
//    int e[3];
//    e[0] = 3;
//    e_ = e;
//
//    fa_[0] = 40;
//    f_ = fa_;
//  }
//  ~Test()
//  {
////    d_.reset();
//    cout << "destruct " << b_ << endl;
//  }
//
//  void print_e() {
//    cout << "in e_[0]: " << e_[0] << endl;
//  }
//
//  void print_f() {
//    cout << "in f_[0]: " << f_[0] << endl;
//  }
//
//  shared_ptr<int> return_d() {
//    return d_;
//  }
//
// public:
//  int a_;
//  string b_;
//  int *c_;
//  shared_ptr<int> d_;
//  int *e_;
//  int fa_[3];
//  int *f_;
//};
//
////int main() {
////
////  {
////    int a = 10;
////    std::shared_ptr<int> ptra = std::make_shared<int>(a);
//////    std::shared_ptr<int> ptra(&a);  // error: munmap_chunk(): invalid pointer
//////    std::shared_ptr<int> ptra(new int(2));
////    std::shared_ptr<int> ptra2(ptra); //copy
////    std::cout << "ptra: " << ptra.use_count() << std::endl;
////    std::cout << "ptra2: " << ptra2.use_count() << std::endl;
////
////    int b = 20;
////    int *pb = &a;
//////    std::shared_ptr<int> ptrb = pb;  //error
////    std::shared_ptr<int> ptrb = std::make_shared<int>(b);
////    ptra2 = ptrb;       // assign
////    pb = ptrb.get();    // get the original pointer
////
////    std::cout << "ptra: " << ptra.use_count() << std::endl;
////    std::cout << "ptra2: " << ptra2.use_count() << std::endl;
////    std::cout << "ptrb: " << ptrb.use_count() << std::endl;
////  }
////
////}
//
//int main() {
//  int *c;
//  shared_ptr<int> d3;
//  shared_ptr<int> d5;
//  shared_ptr<int> d6;
//  int *e, *f;
//
//  {
//    Test t1(1, "t1");
//    Test *t2 = new Test(2, "t2");
//    Test *t3 = new Test(2, "t3");
//    Test t4(1, "t4");
//    Test *t5 = new Test(2, "t5");
//
////    shared_ptr<Test> spt1(&t1);  // error: munmap_chunk(): invalid pointer
//    shared_ptr<Test> spt1 = make_shared<Test>(t1);
//    shared_ptr<Test> spt2(t2);
//    spt2->print_e();
//    spt2->print_f();
//    e = spt2->e_;
//    f = spt2->fa_;
//    cout << "e[0]: " << e[0] << " f[0]: " << f[0] << endl;
////    e = t4.e_;
////    f = t4.fa_;
////    cout << "e[0]: " << e[0] << " f[0]: " << f[0] << endl;
//
//    shared_ptr<Test> spt3 = make_shared<Test>(*t3);
//    c = spt3->c_;
//    cout << "c[0]: " << c[0] << endl;
//
//    cout << "spt3->d_ uc: " << spt3->d_.use_count() << endl;
//    d3 = spt3->d_;
//    cout << "*d3: " << *d3 << endl;
//    // spt3->d_.use_count()==2, one refers to spt3->d_, another refers to spt3 itself
//    cout << "spt3->d_ uc: " << spt3->d_.use_count() << endl;
//    cout << "d3 uc: " << d3.use_count() << endl;
//
//    // t5->d_.use_count()==1, refers to t5->d_
//    cout << "t5->d_ uc: " << t5->d_.use_count() << endl;
//    d5 = t5->d_;
//    cout << "t5->d_ uc: " << t5->d_.use_count() << endl;
//    cout << "d5 uc: " << d5.use_count() << endl;
//
//    shared_ptr<Test> spt6(new Test(2, "t6"));
//    // spt6->d_.use_count()==1, refers to spt6->d_
//    cout << "spt6->d_ uc: " << spt6->d_.use_count() << endl;
//    d6 = spt6->d_;
////    d6 = spt6->return_d();  // the same as d6 = spt6->d_
//    cout << "spt6->d_ uc: " << spt6->d_.use_count() << endl;
//    cout << "d6 uc: " << d6.use_count() << endl;
//
//    delete t3;
//    cout << "spt3->d_ uc: " << spt3->d_.use_count() << endl;
//    cout << "d3 uc: " << d3.use_count() << endl;
//
//    cout << "t5->a: " << t5->a_ << endl;
//    delete t5;
//    // t5->d_.use_count()==1, very odd
//    cout << "t5->d_ uc: " << t5->d_.use_count() << endl;
//    cout << "d5 uc: " << d5.use_count() << endl;
//    cout << "t5->a: " << t5->a_ << endl;
//
//    cout << "{} end" << endl;
//  }
//
////  cout << "spt2: " << spt2.use_count() << std::endl;  // error: ‘spt2’ was not declared in this scope
//  cout << "e[0]: " << e[0] << " f[0]: " << f[0] << endl;
////  delete e;  // error: double free or corruption
////  delete f;  // error: double free or corruption
//
//  cout << "c[0]: " << c[0] << endl;
//  delete c;
//
//  cout << "*d3: " << *d3 << " " << d3.get()[0] << endl;
//  cout << "d3 uc: " << d3.use_count() << endl;
//
//  cout << "*d5: " << *d5 << endl;
//  cout << "d5 uc: " << d5.use_count() << endl;
//  shared_ptr<int> d5_2(d5);
//  cout << "d5 uc: " << d5.use_count() << endl;
//  cout << "d5_2 uc: " << d5_2.use_count() << endl;
//  d5_2.reset();
//  cout << "d5 uc: " << d5.use_count() << endl;
//  cout << "d5_2 uc: " << d5_2.use_count() << endl;
//  d5.reset();
//  cout << "d5 uc: " << d5.use_count() << endl;
//  cout << "d5_2 uc: " << d5_2.use_count() << endl;
//
//  cout << "d6 uc: " << d6.use_count() << endl;
//
//  cout << "main end" << endl;
//
//}
//
//
//
//

int main() {

  return 0;
}
