#include <boost/stacktrace.hpp>
#include <iostream>

int main() {
    const auto trace = boost::stacktrace::stacktrace();
    std::cout << "Boost stacktrace captured " << trace.size() << " frames\n";
    return 0;
}
