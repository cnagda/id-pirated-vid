#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP
#include <type_traits>

template <typename, typename = void>
struct is_pair_iterator : std::false_type { };

template <typename T>
struct is_pair_iterator<T, std::void_t<decltype(std::declval<T>()->first)>>
    : std::true_type { };

template <typename T> 
inline constexpr bool is_pair_iterator_v = is_pair_iterator<T>::value;

#endif