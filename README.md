# modern_cpp_learning

Modern CPP 学习笔记
常量
1. nullptr 为了解决 NULL 存在的函数重载匹配问题，一般情况NULL定义为0， 这样的话无法区分0和空指针；有些编译器将NULL定义为 ((void*)0),但C++不允许void*到其他指针的隐式转换
是nullptr_t类型，可以隐式转换为任意指针类型，也可以用任意指针类型和它做是否相等的比较
2. constexpr 为了提示编译器简化编译期能算出来的表达式， 常量表达可以用于递归函数
3. 【C++17 】if/switch可以声明临时变量，类似go, eg if(auto it = std::find(vec.begin(), vec.end(),3); it != vec.end())*it = 4;
4. 初始化列表即使用“{}”初始化。旧式cpp可对数组，POD（Plain Old Data，即没有构造、析构和虚函数的类或结构体）进行列表初始化,c++11开始可以对带有非POD对象初始化。eg std::vector<vector<int>> v{{12},{13,24}};
5. 结构化绑定,提供了类似多返回值的功能。
6. 类型推导
    变量类型推导 eg auto it = vec.begin();
    入参类型推导[c++20] eg int foo(auto data); // 但是data不能是数组类型
    表达式类型计算。 int x; decltype(x) y; // y被推导为与x相同的类型，即int
    尾返回类型推导
        template<typename T, typename U>
        auto add(T x, U y) ->decltype(x+y){
            return x+y;
        }
        C++14直接auto就行，不需要声明尾返回类型
            template<typename T, typename U>
            auto add(T x, U y) {
                return x+y;
            }
7. foreach迭代 eg for(auto &it: vec){}, 自定义类型实现begin和end方法即可使用forloop
8. 外部模板。防止编译器在每个引用部分进行一次实例化，优化编译性能。extern template class std::vector<double>; // 不在该当前编译文件中实例化模板
9. 类型别名using。可以当typedef用，也能给模板取别名. eg template<typename T, typename U> using is_same_v<T,V>=is_same<T, V>::value;
10. 变长参数模板。
    template<typename... Args> void printf(const std::string &str, Args... args);
    取参数个数sizof...(args)
    参数解包：使用递归+偏特化； 初始化列表展开
    折叠表达式 template<typename...T> auto sum(T... t) {return (t+...);}
11.面向对象
    委托构造。同一个类一个构造函数调用另一个构造函数。
    继承构造。子类不写构造函数，直接使用父类的。using Base::Base; // 继承构造
    显示虚函数重载。override关键字，提示编译器检查该函数是不是重载，如果不是就报错；final关键字，修饰类不能被继承，修饰函数，函数不能被重载。
    构造和析构。 =default 指定使用默认构造 =delete指定删除默认构造，如果调用会编译通不过。
12. lambda表达式
        不捕获，就是个函数对象；捕获的情况就是一个结构体，里面含捕获值状态和一个函数指针
        按值捕获[=]；按引用捕获[&]；表达式捕获,可以在[]中声明新变量并且赋值[v=v1]；泛型lambda：入参可以是auto类型 eg auto add=[](auto x, auto y){return x+y;};

13. 智能指针.shared_ptr, unique_ptr
