#include <climits>
#include <cstddef>
#include <utility>

using namespace std;


struct Element {
    Element(int level, int key, int value): level(level),key(key), value(value) {
        next = new Element*[level];
    }

    
    Element* get_next(int cur_level) {
        return next[cur_level];
    }
    void set_next(Element* e, int cur_level) {
        next[cur_level] = e;
    }
    
    Element** next;
    int value;
    int key; // uniq
    int level;
};


pair<Element*, Element*> find_lower_bound(Element* lhs,  Element* rhs, int key, int cur_level) {
    Element* prev = nullptr;
    while (lhs->key < key) {
        prev = lhs;
        lhs = lhs->get_next(cur_level);
    }
    return {prev, lhs};
}


class Skiplist {
public:
    Skiplist(int max_level) : max_level_(max_level) {
        head_ = new Element(max_level_, INT_MIN, 0);
        tail_ = new Element(max_level_, INT_MAX, 0);
    }
    int get(int key) {
        auto lhs = head_;
        auto rhs = tail_;
        for (int cur_level = max_level_-1; cur_level >= 0; cur_level--) {
            std::tie(lhs, rhs) = find_lower_bound(lhs, rhs, key, cur_level);
            if (rhs->key == key) {
                return rhs->value;
            }
        }
        return -1; 
    }

    void insert(int key, int value) {
        int new_level = get_level();
        auto lhs = head_;
        auto rhs = tail_;
        vector< Element*> prev(new_level, nullptr);
        vector< Element*> next(new_level, nullptr);

        for (int cur_level = max_level_-1; cur_level >= 0; cur_level--) {
            std::tie(lhs, rhs) = find_lower_bound(lhs, rhs, key, cur_level);
            if (rhs->key == key) {
                // return rhs->value;
                rhs->value = value;
                return;
            }
            if (cur_level < new_level) {
                prev[cur_level]= lhs;
                next[cur_level]= rhs; 
            }
        }
        Element* ele = new Element(new_level, key, value);
        for (int i =0 ;i <new_level; i++) {
            prev[i]->set_next(ele, i);
            ele->set_next(next[i], i);
        }
    }
    int get_level() {
        // return rand() % max_level_ + 1;
        int level = 1;
        while(rand() % 2 == 0) {
            level+=1;
            if (level == max_level_) {
                return level;
            }
        }
    }
private:
    int max_level_; // ex. 10,  [0, 10)
    Element* head_;
    Element* tail_;
}