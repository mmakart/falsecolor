#include <random>

static std::mt19937 init_generator()
{
    std::random_device rd;
    return std::mt19937{rd()};
}

template<typename T>
static T random(T start, T end)
{
    static std::mt19937 generator = init_generator();

    std::uniform_int_distribution<T> dist(start, end);
    return dist(generator);
}
