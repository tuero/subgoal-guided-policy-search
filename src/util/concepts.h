// File: concepts.h
// Description: Utility helper for custom concepts amd typetraits

#ifndef HPTS_UTIL_CONCEPTS_H_
#define HPTS_UTIL_CONCEPTS_H_

#include "common/observation.h"

#include <concepts>
#include <functional>
#include <random>
#include <tuple>
#include <type_traits>
#include <variant>

namespace hpts {
namespace detail {

// https://stackoverflow.com/a/53739787/6641216
template <class T, class TypeList>
struct IsContainedIn;

template <class T, class... Ts>
struct IsContainedIn<T, std::variant<Ts...>> : std::bool_constant<(... || std::is_same<T, Ts>{})> {};

// https://stackoverflow.com/a/54191646
template <typename T, template <typename...> class Z>
struct is_specialization_of : std::false_type {};

template <typename... Args, template <typename...> class Z>
struct is_specialization_of<Z<Args...>, Z> : std::true_type {};

template <typename T, template <typename...> class Z>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Z>::value;

template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

// https://stackoverflow.com/a/71032888/6641216
template <typename T>
struct template_parameter;

template <template <typename...> class C, typename T>
struct template_parameter<C<T>> {
    using type = T;
};

template <typename T, typename... Ts>
struct contains : std::bool_constant<(std::is_same<T, Ts>{} || ...)> {};

}    // namespace detail

/**
 * Extract inner template type
 */
template <typename T>
using template_parameter_t = typename detail::template_parameter<T>::type;

/**
 * Dummy value to use for concepts
 */
template <class T>
typename std::add_rvalue_reference<T>::type makeval();

/**
 * Concept for type T is among a given variants accepted types
 */
template <typename T, typename U>
concept IsTypeAmongVariant = detail::IsContainedIn<T, U>::value;

/**
 * Check for if T is a template specialization of Z
 */
template <typename T, template <typename...> class Z>
concept IsSpecialization = detail::is_specialization_of_v<T, Z>;

/**
 * Static switch on index over collection of types
 * usage: using type = static_switch<0, T, U>
 * https://stackoverflow.com/a/15404153/6641216
 */
template <std::size_t N, typename... T>
using static_switch = typename std::tuple_element<N, std::tuple<T...>>::type;

/**
 * Allow for static_assert trigger in else block of if constexpr
 * usage: static_assert(bool_value<false, TypeOfValue>::value, "Unsupported
 * type"); https://stackoverflow.com/q/62359172/6641216
 */
template <bool v, typename T>
struct bool_value {
    static constexpr bool value = v;
};

// enable_bitmask_operator_or must be set on scopped Enum to allow for typed bitwise
template <typename T>
concept IsEnumBitMaskEnabled = std::is_enum_v<T> && requires(T e) { enable_bitmask_operators(e); };

template <IsEnumBitMaskEnabled T>
constexpr auto operator|(const T lhs, const T rhs) {
    using underlying = std::underlying_type_t<T>;
    return static_cast<T>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}
template <IsEnumBitMaskEnabled T>
constexpr auto operator&(const T lhs, const T rhs) {
    using underlying = std::underlying_type_t<T>;
    return static_cast<T>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}

/**
 * Concept to check if a type T is any of the given types in Ts ...
 */
template <typename T, typename... Ts>
concept IsAny = detail::contains<T, Ts...>::value;

/**
 * Concept to check if a type specializes std hash
 * https://stackoverflow.com/a/51915825/6641216
 */
template <typename T>
concept IsSTDHashable = detail::is_std_hashable_v<T>;

/**
 * Concept to check if a type has a heuristic
 */
template <typename T>
concept HasHeuristic = requires(T t) {
    {
        t.heuristic
    } -> std::same_as<double &>;
};

/**
 * Concept to check if a type has a policy
 */
template <typename T>
concept HasPolicy = requires(T t) {
    {
        t.policy
    } -> std::same_as<std::vector<double> &>;
};

/**
 * Concept to check if a type as an observation
 */
template <typename T>
concept HasObservation = requires(T t) {
    {
        t.observation
    } -> std::same_as<Observation &>;
};

/**
 * Concept to check if a type has a random generator
 */
template <typename T>
concept HasRNG = requires(T t) {
    {
        t.rng
    } -> std::same_as<std::mt19937 &>;
};
}    // namespace hpts

#endif    // HPTS_UTIL_CONCEPTS_H_
