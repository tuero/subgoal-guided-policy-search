// File: block_allocator.h
// Description: Block allocator with stable pointer addresses

#ifndef HPTS_UTIL_BLOCK_ALLOCATOR_H_
#define HPTS_UTIL_BLOCK_ALLOCATOR_H_

#include <spdlog/spdlog.h>

#include <absl/container/flat_hash_set.h>

#include <cassert>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace hpts {

template <typename T, typename HashT = std::hash<T>, typename KeyEqualT = std::equal_to<T>>
class BlockAllocator {
public:
    BlockAllocator() = delete;
    BlockAllocator(int allocate_increment, std::optional<T> default_item = std::nullopt)
        : allocate_increment_(allocate_increment), default_item_(std::move(default_item)) {
        if (allocate_increment_ < 1) {
            SPDLOG_ERROR("allocate_increment must be >= 1");
            std::exit(1);
        }
        clear();
    }

    /**
     * Add to the container and receive a pointer to it
     * @note if the item is already stored, a pointer to the previously stored item is returned
     * @param item The item to store
     * @return A pointer to the stored item
     */
    template <typename U>
    auto add(U &&item) -> const T * {
        assert(idx < static_cast<int>(blocks.back().size()) - 1);
        if (has_item(item)) {
            return get_ptr(item);
        }

        blocks.back()[static_cast<std::size_t>(++idx)] = std::forward<U>(item);
        const auto item_ptr = &(blocks.back()[static_cast<std::size_t>(idx)]);
        items_set.insert(item_ptr);

        if (idx >= allocate_increment_ - 1) {
            add_block();
        }
        return item_ptr;
    }

    /**
     * Check if the given state is already being held
     * @param item State to check
     * @return True if the given state is already held, false otherwise
     */
    [[nodiscard]] auto has_item(const T &item) const -> bool {
        return items_set.find(&item) != items_set.end();
    }

    /**
     * Get the pointer to the held state which matches the state being queried
     * @param item State to check
     * @return Pointer to held state if exists, nullptr otherwise
     */
    [[nodiscard]] auto get_ptr(const T &item) const -> const T * {
        const auto itr = items_set.find(&item);
        return (itr == items_set.end()) ? nullptr : *itr;
    }

    /**
     * Clear all held items and reset the state
     */
    void clear() {
        idx = -1;
        items_set.clear();
        for (auto &block : blocks) {
            block.clear();
        }
        blocks.clear();
        add_block();
    }

private:
    void add_block() {
        if constexpr (std::is_default_constructible_v<T>) {
            if (default_item_) {
                blocks.push_back(std::vector<T>(allocate_increment_, default_item_.value()));
            } else {
                blocks.push_back(std::vector<T>(allocate_increment_));
            }
        } else {
            blocks.push_back(std::vector<T>(static_cast<std::size_t>(allocate_increment_), default_item_.value()));
        }
        idx = -1;
    }

    // Required because T may have hash defined, but pointers are stored
    class ItemPtrHash {
    public:
        auto operator()(const T *item) const -> std::size_t {
            return hasher(*item);
        }

    private:
        HashT hasher;
    };

    // Required because T may have equal_to defined, but pointers are stored
    class ItemPtrEqual {
    public:
        auto operator()(const T *left, const T *right) const -> bool {
            return equal_to(*left, *right);
        }

    private:
        KeyEqualT equal_to;
    };

    using ItemSet = absl::flat_hash_set<const T *, ItemPtrHash, ItemPtrEqual>;

    int allocate_increment_;
    int idx = -1;
    std::optional<T> default_item_;
    std::vector<std::vector<T>> blocks;
    ItemSet items_set;
};

}    // namespace hpts

#endif    // HPTS_UTIL_BLOCK_ALLOCATOR_H_
