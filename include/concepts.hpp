#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP
#include <type_traits>

template <typename, typename = void, typename = void>
struct is_pair_iterator : std::false_type
{
};

template <typename T>
struct is_pair_iterator<T,
                        std::void_t<decltype(std::declval<T>()->first)>,
                        std::void_t<decltype(std::declval<T>()->second)>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_pair_iterator_v = is_pair_iterator<T>::value;

template <typename, typename = void>
struct is_frame_iterator : std::false_type
{
};

template <typename It>
struct is_frame_iterator<It,
                         std::void_t<decltype(std::declval<It>()->descriptors)>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_frame_iterator_v = is_frame_iterator<T>::value;

template <typename T, typename = void>
struct has_arrow : std::false_type {};
                   template <typename T>
                   struct has_arrow<T, std::void_t<decltype(std::declval<T>().operator->())>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool has_arrow_v = has_arrow<T>::value;

#endif