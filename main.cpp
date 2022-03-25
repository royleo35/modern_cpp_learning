#include <iostream>
#include <type_traits>
#include <vector>
#include <initializer_list> // initializer_list
#include <tuple>
#include <list>
#include <map>
#include <functional>
#include <regex>
#include <thread>
#include <mutex>
#include <chrono>
#include <future>
#include <queue>
#include <condition_variable>
#include <atomic>

#define p(x) std::cout<<(#x)<<":"<<x<<std::endl;

void foo_nullptr(char* ){
    std::cout<<"char* called"<<std::endl;
}
void foo_nullptr(int){
    std::cout<<"int called"<<std::endl;
}


void test_nullptr(){
    if (std::is_same_v<decltype(NULL), decltype(0)>) {
        std::cout<<"NULL==0"<<std::endl;
    }else if (std::is_same_v<decltype(NULL), decltype((void*)0)>) {
        std::cout<<"NULL== (void*)0"<<std::endl;
    }else if (std::is_same_v<decltype(NULL), std::nullptr_t>) {
        std::cout<< "NLL==nullptr"<<std::endl;
    }
    foo_nullptr((char*)NULL);
    foo_nullptr(nullptr);
    foo_nullptr(0);
    foo_nullptr((char*)(void*)0);
    
}

#define LEN 10

int len_foo(){
    int i = 2;
    return i;
}

constexpr int len_foo_constexpr(){
    return 5;
}

constexpr int febo(int n) {
    if(n<0) return 0;
    if(n <2) return n;
    return febo(n-1) + febo(n-2);
}

void test_constexpr(){
    int a[len_foo_constexpr()];
    std::cout<<febo(10)<<std::endl;
}

class InitFoo{
public:
    std::vector<int> vec;
    // 提供初始化列表构造函数
    InitFoo(std::initializer_list<int> list) {
        for(auto& it: list){
            vec.push_back(it);
        }
    }
};

void foo_init(std::initializer_list<int> list){
    std::cout<<*list.begin()<<std::endl;
}

std::tuple<int, float, std::string> foo_con_bind() {
    return std::make_tuple(1, 0, "hello");
}

struct Name {
    int id;
    float money;
    std::string name;
};

void test_construction_bind() {
    auto [id, money, name] = foo_con_bind();
    auto [id1, m1, name1] = Name{10, 0, "hah"};
    std::cout<<name<<"\t"<<name1<<std::endl;
}

void test_initialization_list(){
    std::vector<std::vector<int>> v{{1,2}, {2,3}};
    std::cout<<v[0][1]<<std::endl;
    InitFoo initFoo{1,2,3};
    std::cout<<initFoo.vec[1]<<std::endl;
    foo_init({1,2});

}

// c++11
template<typename T, typename U>
auto foo_tail1(T x, U y) -> decltype(x+y) {
    return x+y;
}

//c++14
template<typename T, typename U>
auto foo_tail2(T x, U y) {
    return x+y;
}


void foo_type_calc(auto x){
    std::cout<<x<<std::endl;
}


template<typename T>
decltype(auto) foo_decl_auto(T&& val) {
    std::cout<<"is value type:"<<!std::is_reference_v<decltype(val)><<std::endl;
    std::cout<<"is ref type:"<<std::is_reference_v<decltype(val)><<std::endl;
    std::cout<<"is left ref type:"<<std::is_lvalue_reference_v<decltype(val)><<std::endl;
    std::cout<<"is right ref type:"<<std::is_rvalue_reference_v<decltype(val)><<std::endl;
    std::cout<<"\n\n";
}

void test_type_calc(){
    // 参数类型推导
    auto y = "sdafs";
    std::cout<<y<<std::endl;
    // 入参类型推导
    foo_type_calc(1);
    foo_type_calc("1313");
    foo_type_calc(std::string("31"));
    // 表达式类型推导 e 推导为与x相同类型
    int x = 10;
    decltype(x) e=1000;
    std::cout<<e<<std::endl;

    //尾返回类型推导
    foo_tail1(1, 1.0);
    foo_tail2(2, 4);

    // decltype(auto) 参数转发 主要用于完美转发左值和右值
    foo_decl_auto(x);
    foo_decl_auto(10);
    int& lref = x;
    foo_decl_auto(lref);
    int&& rref = 11;
    foo_decl_auto(rref);
    // cont left ref
    const int& clref = 100;
    foo_decl_auto(clref);

}

// 编译时能判断if分支，会根据调用情况拆分成3个函数
template<typename T>
void foo_print_if_const(const T& val) {
    if constexpr (std::is_same_v<float, T>) {
        std::cout<<"float"<<std::endl;
    } else if (std::is_integral_v<T>) {
        std::cout<<"integral"<<std::endl;
    } else {
        std::cout<<"unknown type"<<std::endl;
    }
}

void test_if_constexpr(){
    foo_print_if_const(1.0);
    foo_print_if_const(float(1.1));
    foo_print_if_const(1);
}

template<typename T, size_t N>
class MyArray{
private:
    T data[N];
public:
    size_t pos;
    MyArray(std::initializer_list<T> list){
        pos=0;
        if(list.size() == 0) {
            abort();
        }
        int i = 0;
        auto it = list.begin();
        for(;i < N && it != list.end(); ++it, ++i) {
            data[i] = *it;
        }
    }
    T* begin(){
        std::cout<<"begin\n";
        return data;
    }
    T* end() {
        std::cout<<"end\n";
        return data + N;
    }
    T* operator++(){
        std::cout<<"operator++\n";
        pos++;
        return data+pos;
    }
    MyArray operator++(int) {
        MyArray tmp(*this);
        ++(*this);
        return tmp;

    }


};

void test_for_each() {
    std::vector<int> v{1,2,3};
    for(auto i: v){
        std::cout<<i<<std::endl;
    }
    for(auto& i: v){
        i += 1;
    }
    for(auto i: v){
        std::cout<<i<<std::endl;
    }

    MyArray<int, 3> a{4,5,6};
    auto it = a.begin();
    ++it;
    for(auto i: a){
        std::cout<<i<<std::endl;
    }

}

// 递归+偏特化解包模板参数
template<typename T, typename ... Ts>
void my_print(T val, Ts... args) {
    std::cout<<val<<std::endl; // 基本情况 相当于是特化情况，这样不用写两个模板
    if constexpr (sizeof ...(args) > 0) my_print(args...); // 递归调用
}


template<typename T, typename ... Ts>
void magic_print(T val, Ts... args) {
    std::cout<<val<<std::endl; // 基本情况 相当于是特化情况，这样不用写两个模板
    // 使用初始化列表+逗号表达式+lambda
    (void) std::initializer_list<T>{([&args](){std::cout<<args<<std::endl;}(), val)...};
}

// 测试逗号表达式
void test_comma(){
    int x = 10;
    int tmp = x;
    // x^4 +x^2 + 1
    int res = (x=x*x, x*x + x + 1);
    int want = tmp*tmp*tmp*tmp + tmp*tmp + 1;
    assert(res == want);
    std::cout<<res<<std::endl;
}


void test_arg_template() {
    my_print(1, 1.2, "23124", std::string("heh"));
    my_print("1");
    magic_print(1, 1.2, "23124", std::string("heh"));

}

template<typename... T>
auto clo_expr(T... t){
    return (t+...);
}

void test_clo_expr() {
    std::cout<<clo_expr(1.0, 1.0, 7.3)<<std::endl;
    std::cout<<clo_expr(std::string("13"), std::string("hehe\n"));
}

// 非类型模板参数推导
template<int n>
void foo_N() {
    std::cout<<n<<std::endl;
}

// c++17 支持使用auto，但实例化的参数n只能是整形字面量，不能是type
template<auto n>
void foo_autoN() {
    std::cout<<n<<std::endl;
}



void test_foo_N(){
    foo_N<10>();
    foo_N<100>();

    foo_autoN<101>();
    foo_autoN<char(65)>();

    const long N = 3;
    int a[N];
    std::cout<<a[2]<<std::endl;
    foo_autoN<N>();
}

// 委托(delegate)构造，类中一个构造函数可以调用另一个构造函数

class Delegate{
public:
    Delegate(){
        val1 = 10;
    }
    Delegate(int val2): Delegate() {
        this->val2 = val2;
    }
    void print(){
        std::cout<<val1+val2<<std::endl;
    }
private:
    int val1,val2;
};
void test_delegate_constructor(){
    Delegate d(30);
    d.print();
}

// 继承构造
class SubClass: public Delegate{
public:
    using Delegate::Delegate; // c++11 引入了继承构造
};


void test_inherit_constructor(){
    SubClass s(10); // 继承构造，调用的Delegate::Delegate(int val2)
    s.print();
}

// override final
class Base{
    virtual int foo() final;
    // int foo2() final; // 非虚函数不能标识为final
};

class Derived final : public Base{ // Derived 不能被继承
//    virtual int foo() override{ // final的虚函数不能被重写
//
//    }
};

// exercize 1
// 使用结构化绑定
template <typename Key, typename Value, typename F>
void update(std::map<Key, Value>& m, F foo) {
    for(auto&& [k, v]: m) {v=foo(k);}
}
void test_con_bind_map() {
    std::map<std::string, long long int> m {
            {"a", 1},
            {"b", 2},
            {"c", 3}
    };
    update(m, [](std::string key) {
        return std::hash<std::string>{}(key);
    });
    for (auto&& [key, value] : m)
        std::cout << key << ":" << value << std::endl;
}

template<typename... T>
auto mean(T... ts)  {
    if (sizeof...(ts) == 0) {
        abort();
    }
    return (ts+...) / sizeof...(ts);
}

void test_mean() {
    std::cout<<mean(1, 3)<<std::endl;
    std::cout<<mean(1, 4.0)<<std::endl;
    std::cout<<mean(5.8, 4.0)<<std::endl;
}

// lambda
void test_lambda(){
    // 不带捕获
    auto f1 = [](){std::cout<<"f1"<<std::endl;};
    f1();
    // 按值捕获
    int val = 10;
    auto f2 = [=](){return val;};
    val = 11;
    std::cout<<f2()<<std::endl;

    // 按引用捕获
    auto f3 = [&](){return val;};
    val = 12;
    std::cout<<f3()<<std::endl;

    // 表达式捕获，[]里面可以加变量初始化语句，也能捕获右值了
    const int vv = 1434;
    auto f4 = [v1 = 1000, v2 = vv](int x){
        std::cout<<v1+v2+x<<std::endl;
    };
    f4(123);

    // 泛型lambda, 即lambda的参数可以是auto，让编译器推断
    auto add = [](auto x, auto y){return x + y ;};
    std::cout<<add(1, 2)<<std::endl;
    auto res = add(1, 43.0);
    std::cout<<res<<std::endl;
    std::cout<<std::is_same_v<decltype(res), float><<std::endl;
    std::cout<<std::is_same_v<decltype(res), double><<std::endl; // 浮点数字面量默认推断成double

}

auto g_add = [](auto x, auto y) {return x + y ;};
// functional bind
void test_functional(){
    // functional 是一个函数容器，可以容纳所有类型的可调用对象
    std::function<void(int)> f = [](int){};
    int val = 10;
    std::function<int(int)> f2 = [&](int v){return val + v;};

    // std::bind 可以先固定可调用对象的一些参数，使用std::placeholder占位(占位符从1开始)，等传入剩余参数再调用
    auto ff = [](int x, int y, int z){return x*y*z;};
    auto pp = std::bind(ff, 10, std::placeholders::_1, 21);
    std::cout<<pp(3)<<std::endl;

    auto fff = [](int x, int y, int z){std::cout<<x<<"\t"<<y<<"\t"<<z<<std::endl;};
    // 占位符可以乱序，下面调用是会先传入z, 再传入x
    auto ppp = std::bind(fff, std::placeholders::_2, 10, std::placeholders::_1);
    ppp(1, 2);

    // 逆转参数顺序
    auto p4 = std::bind(fff, std::placeholders::_3, std::placeholders::_2, std::placeholders::_1);
    p4(1, 2, 3);

    // 测试lambda能否全局调用
    g_add(1, 10);
}


template<typename T, typename ... Args>
auto make_unique(Args&&... args){
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

struct PtrA;
struct PtrB;

struct PtrA{
    std::shared_ptr<PtrB> pb;
    ~PtrA() {
        std::cout<<"de con PtrA\n";
    }
};

struct PtrB{
    std::shared_ptr<PtrA> pa;
    ~PtrB() {
        std::cout<<"de con PtrB\n";
    }
};


struct WeakA;
struct WeakB;

struct WeakA{
    std::shared_ptr<WeakB> pb;
    ~WeakA() {
        std::cout<<"de con WeakA\n";
    }
};

struct WeakB{
    std::weak_ptr<WeakA> pa;
    ~WeakB() {
        std::cout<<"de con WeakB\n";
    }
};

void test_weak_ptr(){
    // 用途1： 解决循环引用问题
    // weak_ptr 弱引用，解决循环引用的问题
    {
        auto pa = std::make_shared<PtrA>(); // pa ref = 1
        auto pb = std::make_shared<PtrB>(); // pb ref = 1
        pa->pb = pb; // pb ref = 2
        pb->pa = pa; // pa ref = 2
        assert(pa.use_count() == pb.use_count() );
    }// a 和 b对象的引用计数都是1，析构函数都没有被调用，解决方式，其中一个类型由shared_ptr 改为 weak_ptr

    {
        auto pa = std::make_shared<WeakA>(); // pa ref = 1
        auto pb = std::make_shared<WeakB>(); // pb ref = 1
        pa->pb = pb; // pb ref ++= 2
        pb->pa = pa; // pa ref 不变，还是为1，因为pb->pa 类型是weak_ptr
        assert(pa.use_count() == 1 );
        assert(pb.use_count() == 2);
    }
    // 离开作用域时 pa ref -- = 0 WeakA析构被调用, pb ref -1 = 1
    // WeakA的析构被调用之后也会调用WeakA成员pb的析构，进而将pb的引用计数也-1 = 0， 导致WeakB的析构被调用


    // 用途2：解决悬垂指针问题
    // 旧式方法, ptr 释放后将导致ref指向不可预知的数据
    int *ptr = new int(10);
    int *ref = ptr;
    delete ptr;

    // 空对象
    std::shared_ptr<int> sptr;

    sptr.reset(new int);
    *sptr = 1;

    std::weak_ptr<int> w1 = sptr;

    sptr.reset(new int); // 对象1 将被释放， w1 dangling
    *sptr = 2;

    if(w1.expired()){//  go this branch
        std::cout<<"w1.expired()\n";
    } else{
        auto tmp = w1.lock();
        std::cout<<"val1:"<<*tmp<<std::endl;
    }

    std::weak_ptr<int> w2 = sptr; // 指向对象2
    if(w2.expired()){
        std::cout<<"w2.expired()\n";
    } else{//  go this branch
        auto tmp = w2.lock();
        std::cout<<"val2:"<<*tmp<<std::endl;
    }

}

void test_shared_ptr(){
    // shared_ptr
    auto p =  std::make_shared<int>(10); // ref cnt 1
    auto& p2 = p; // ref cnt = 1 p2 is ref of p, ref cnt not change
    auto p3 = p;  // ref cnt ++ = 3
    std::cout<<*p.get()<<std::endl;  // get() method to get raw pointer to obj
    std::cout<<p.use_count()<<std::endl; // use_count() method get ref cnt, will not change ref cnt
    assert(*p.get() == 10);
    assert(p.use_count() == 2);
    // p2 reset 之后 p2.ptr==nullptr, p.ref=p2.ref=1
    p2.reset(); // p.ref = p2.ref  =0 p3.ref=1
    assert(p.use_count() == 0);
    assert(p3.use_count() == 1);
    auto pref = p.use_count();
    auto p2ref = p2.use_count();
    assert(pref == p2ref);
    {
        auto p4 = p3; // ref++ = 2
        assert(p4.use_count() == 2);
        *p4 = 20;
        assert(*p3.get() == 20);

    } // p4 de constructor cnt-- = 1
    assert(p3.use_count()==1);
    p2 = p3; // p2 重新指向p3 ref++=2
    assert(p2.use_count() == 2);
    assert(p2.use_count() == p3.use_count());
}

void test_unique_ptr(){
    // unique_ptr 独占的，不能拷贝，可以move, move 之后，源对象指向nullptr
    //auto pu = std::unique_ptr<int>(new int(10));
    auto pu = make_unique<int>(10);
    assert(*pu.get() == 10);
    *pu = 20;
    // auto pu1 = pu; 不能编译通过
    auto  pu1 = std::move(pu);
    assert(*pu1.get() == 20);
    assert(pu == nullptr);
}



void test_smart_pointer(){
   test_shared_ptr();
   test_unique_ptr();
   test_weak_ptr();
}


void test_reg1(){// 匹配小写字母为名字的txt文件名
    std::string fnames[] = {"foo.txt", "bar.txt", "test", "a0.txt", "AAA.txt"};
    std::regex txt_reg("[a-z]+\\.txt");
    for(const auto& fname: fnames){
        auto match = std::regex_match(fname, txt_reg);
        std::cout<<fname<<":"<<match<<std::endl;
    }
}

void test_reg2(){
    std::string fnames[] = {"foo.txt", "bar.txt", "test", "a0.txt", "AAA.txt"};
    std::regex base_reg("([a-z]+)\\.txt");
    std::smatch res;
    for(const auto& fname: fnames){
        auto match = std::regex_match(fname, res, base_reg);
        if (match && res.size() == 2) {
            // res[0] 表示整个串
            // res[1]表示匹配到的小括号里的内容
            std::cout<<fname<<": full->"<<res[0].str()<<";match_str->"<<res[1].str()<<std::endl;
        }


    }
}
void test_regex(){
    test_reg1();
    test_reg2();
}

void test_thread(){
    std::thread t([](){std::cout<<"Hello\n";});
    t.join();
}

void test_unique_lock(){
    // unique_lock 是对lock_guard的优化，独占资源（不能拷贝，可以移动），并且可以显示调用lock和unlock，比较灵活
    int val = 1;
    auto f = [&val](int new_val){
        static std::mutex mu;
        std::unique_lock<std::mutex> lock(mu);
        // 这里不需要lock，构造的时候就会调用lock lock.lock();
        int old_val = val;
        std::cout<<"val from:"<<val<<" changed to:"<<new_val<<std::endl;
        val = new_val;
        std::cout<<"verify new_val:"<<val<<std::endl;
        lock.unlock();

        std::cout<<"split---------\n";
        // 再把值改回去
        lock.lock();
        std::cout<<"val from:"<<val<<" changed to:"<<old_val<<std::endl;
        val = new_val;
        std::cout<<"verify new_val:"<<val<<std::endl;
        // lock.unlock(); 这里也不需要显示unlock 结束时unique_lock的析构会调用unlock

    };
    std::thread t1(f, 2), t2(f, 3);
    t1.join();
    t2.join();
}

void test_mutex(){
    // 1 使用裸mutex
    int val = 1;
    auto f = [&val](int new_val){
        static std::mutex mu;
        mu.lock();
        std::cout<<"val from:"<<val<<" changed to:"<<new_val<<std::endl;
        val = new_val;
        std::cout<<"verify new_val:"<<val<<std::endl;
        mu.unlock();
    };
    std::thread t1(f, 2), t2(f, 3);
    t1.join();
    t2.join();

    // 2. 使用lock_guard 在析构的时候unlock, 防止lock和unlock中途抛异常
    auto fg = [&val](int new_val){
        static std::mutex mu;
        int tmp = val;
        {
            std::lock_guard<std::mutex> lock(mu);
            std::cout<<"val from:"<<val<<" changed to:"<<new_val<<std::endl;
            val = new_val;
            std::cout<<"verify new_val:"<<val<<std::endl;

        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        { // 再交换回来
                std::lock_guard<std::mutex> lock(mu);
                std::cout<<"val from:"<<val<<" changed to:"<<tmp<<std::endl;
                val = tmp;
                std::cout<<"verify new_val:"<<val<<std::endl;

        }
    };
    std::thread t3(fg, 4), t4(fg, 6);
    t3.join();
    t4.join();

}

void test_future(){
    std::packaged_task<int(int,int)> task([](int a, int b){
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return a + b;
    });
    std::future<int> res = task.get_future();
    std::thread(std::move(task), 10, 3234).detach(); // 开一个线程执行任务
    res.wait(); // 等待任务执行完成
    int val = res.get();
    std::cout<<val<<std::endl;
}

void test_condition_variable(){
    // 利用条件变量实现生产者消费者模型
    std::queue<int> produced_nums;
    std::mutex mu;
    std::condition_variable cv;
    bool notified = false;
    const int NUM = 10;

    // producer
    auto producer = [&](){
        for(int i=0; i < NUM ; i++){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::unique_lock<std::mutex> lock(mu);
            std::cout<<"producing:"<<i<<std::endl;
            produced_nums.push(i);
            notified = true;
            cv.notify_all();
        }
    };
    // consumer
    auto consumer = [&](){
        int cnt = 0;
       while(cnt< NUM){
           std::unique_lock<std::mutex> lock(mu);
           while(!notified){// 避免虚假唤醒
               cv.wait(lock);
           }
           lock.unlock();
           std::this_thread::sleep_for(std::chrono::microseconds(50)); // 消费速度慢于生产速度
           lock.lock();
           while(!produced_nums.empty()){
               std::cout<<"consuming:"<<produced_nums.front()<<std::endl;
               produced_nums.pop();
           }
           notified = false;
           cnt++;
       }
    };

    std::thread pp(producer); // 生产者线程
    std::thread cs[2]; // 消费者线程
    for(auto & c : cs){
        c = std::thread(consumer);
    }
    pp.join();
    for(auto &c: cs){
        c.join();
    }
}

void test_concurrency(){
    test_thread();
    test_mutex();
    test_unique_lock();
    test_condition_variable();
}


struct A {
    float x;
    int y;
    long long z;
};

void test_atomic(){
    std::atomic<int> cnt{1};
    std::thread t1([&cnt]{cnt.fetch_add(-10);});
    std::thread t2([&cnt]{
        cnt ++;
        cnt += 20;
    });
    t1.join();
    t2.join();
    p(cnt);

    // 判断类型是否支持原子操作
    std::atomic<A> a;
    p(a.is_lock_free());
}

void test_raw_str(){
    // 使用R + () 来实现原始字符串字面量，这样括号里面的内容不需要转义
    std::string s = R"("Hello")"; // 里面的双引号不需要转义
    p(s);
}

// 字符串字面量自定义必须设置如下的参数列表
std::string operator"" _wow1(const char *wow1, size_t len) {
    return std::string(wow1)+"woooooooooow, amazing";
}

std::string operator"" _wow2 (unsigned long long i) {
    return std::to_string(i)+"woooooooooow, amazing";
}

// 原始字符串重载
void test_raw_str_overload() {
    auto str = "abc"_wow1; // <==> operator""_wow1("abc", 4);
    auto num = 1_wow2;  // <===> operator""_wow2(1);
    std::cout << str << std::endl;
    std::cout << num << std::endl;
    const char * s = "haha";
    std::cout << operator""_wow1(s, 0);
}

struct alignas(std::max_align_t) B {
    char a;
    short b;
    char c;
};
void test_align(){
    p(alignof(A));
    p(alignof(B));
}

int main() {
    test_nullptr();
    test_constexpr();
    test_initialization_list();
    test_construction_bind();
    test_type_calc();
    test_if_constexpr();
    test_for_each();
    test_arg_template();
    test_comma();
    test_clo_expr();
    test_foo_N();
    test_delegate_constructor();
    test_inherit_constructor();
    test_con_bind_map();
    test_mean();
    test_lambda();
    test_functional();
    test_smart_pointer();
    test_regex();
    test_concurrency();
    test_future();
    test_atomic();
    test_raw_str();
    test_raw_str_overload();
    test_align();
    return 0;
}
