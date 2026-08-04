// redis_zskiplist.h - A faithful port of Redis's zskiplist (from t_zset.c)
//
// This is a copy of the skiplist used by Redis sorted sets, adapted to be
// self-contained C++. The algorithm and data structure are identical to
// Redis 7.x src/t_zset.c; only the dependencies were swapped:
//   - sds          -> std::string
//   - zmalloc      -> operator new / delete
//   - serverAssert -> assert
//   - random()     -> std::rand()
//
// Ported functions: zslCreate / zslFree, zslInsert, zslDelete, zslUpdateScore,
// zslGetRank, zslGetElementByRank, zslIsInRange, zslFirstInRange.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>

#define ZSKIPLIST_MAXLEVEL 32 /* Should be enough for 2^64 elements */
#define ZSKIPLIST_P 0.25      /* Skiplist P = 1/4 */

typedef struct zskiplistNode {
    std::string ele;
    double score;
    struct zskiplistNode *backward;
    struct zskiplistLevel {
        struct zskiplistNode *forward;
        unsigned long span;
    } level[];
} zskiplistNode;

typedef struct zskiplist {
    struct zskiplistNode *header, *tail;
    unsigned long length;
    int level;
} zskiplist;

typedef struct zrangespec {
    double min, max;
    int minex, maxex;
} zrangespec;

/* Create a skiplist node with the specified number of levels.
 * 'ele' is copied into the node after the call. */
static zskiplistNode *zslCreateNode(int level, double score, const std::string &ele) {
    zskiplistNode *zn = (zskiplistNode *)malloc(sizeof(*zn) + level * sizeof(zskiplistNode::zskiplistLevel));
    zn->score = score;
    new (&zn->ele) std::string(ele);
    return zn;
}

/* Create a new skiplist. */
static zskiplist *zslCreate(void) {
    int j;
    zskiplist *zsl = (zskiplist *)malloc(sizeof(*zsl));

    zsl->level = 1;
    zsl->length = 0;
    zsl->header = zslCreateNode(ZSKIPLIST_MAXLEVEL, 0, std::string());
    for (j = 0; j < ZSKIPLIST_MAXLEVEL; j++) {
        zsl->header->level[j].forward = NULL;
        zsl->header->level[j].span = 0;
    }
    zsl->header->backward = NULL;
    zsl->tail = NULL;
    return zsl;
}

/* Free the specified skiplist node. The referenced std::string is freed too. */
static void zslFreeNode(zskiplistNode *node) {
    node->ele.~basic_string();
    free(node);
}

/* Free a whole skiplist. */
static void zslFree(zskiplist *zsl) {
    zskiplistNode *node = zsl->header->level[0].forward, *next;

    zslFreeNode(zsl->header);
    while (node) {
        next = node->level[0].forward;
        zslFreeNode(node);
        node = next;
    }
    free(zsl);
}

/* Returns a random level for the new skiplist node we are going to create.
 * The return value of this function is between 1 and ZSKIPLIST_MAXLEVEL
 * (both inclusive), with a powerlaw-alike distribution where higher
 * levels are less likely to be returned. */
static int zslRandomLevel(void) {
    static const int threshold = ZSKIPLIST_P * RAND_MAX;
    int level = 1;
    while (rand() < threshold)
        level += 1;
    return (level < ZSKIPLIST_MAXLEVEL) ? level : ZSKIPLIST_MAXLEVEL;
}

/* Insert a new node in the skiplist. Assumes the element does not already
 * exist (up to the caller to enforce that). The skiplist takes ownership
 * of the passed std::string 'ele'. */
static zskiplistNode *zslInsert(zskiplist *zsl, double score, const std::string &ele) {
    zskiplistNode *update[ZSKIPLIST_MAXLEVEL], *x;
    unsigned long rank[ZSKIPLIST_MAXLEVEL];
    int i, level;

    assert(!std::isnan(score));
    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        /* store rank that is crossed to reach the insert position */
        rank[i] = i == (zsl->level - 1) ? 0 : rank[i + 1];
        while (x->level[i].forward &&
               (x->level[i].forward->score < score ||
                (x->level[i].forward->score == score &&
                 x->level[i].forward->ele < ele))) {
            rank[i] += x->level[i].span;
            x = x->level[i].forward;
        }
        update[i] = x;
    }
    /* we assume the element is not already inside, since we allow duplicated
     * scores, reinserting the same element should never happen since the
     * caller of zslInsert() should test in the hash table if the element is
     * already inside or not. */
    level = zslRandomLevel();
    if (level > zsl->level) {
        for (i = zsl->level; i < level; i++) {
            rank[i] = 0;
            update[i] = zsl->header;
            update[i]->level[i].span = zsl->length;
        }
        zsl->level = level;
    }
    x = zslCreateNode(level, score, ele);
    for (i = 0; i < level; i++) {
        x->level[i].forward = update[i]->level[i].forward;
        update[i]->level[i].forward = x;

        /* update span covered by update[i] as x is inserted here */
        x->level[i].span = update[i]->level[i].span - (rank[0] - rank[i]);
        update[i]->level[i].span = (rank[0] - rank[i]) + 1;
    }

    /* increment span for untouched levels */
    for (i = level; i < zsl->level; i++) {
        update[i]->level[i].span++;
    }

    x->backward = (update[0] == zsl->header) ? NULL : update[0];
    if (x->level[0].forward)
        x->level[0].forward->backward = x;
    else
        zsl->tail = x;
    zsl->length++;
    return x;
}

/* Internal function used by zslDelete, zslDeleteRangeByScore and
 * zslDeleteRangeByRank. */
static void zslDeleteNode(zskiplist *zsl, zskiplistNode *x, zskiplistNode **update) {
    int i;
    for (i = 0; i < zsl->level; i++) {
        if (update[i]->level[i].forward == x) {
            update[i]->level[i].span += x->level[i].span - 1;
            update[i]->level[i].forward = x->level[i].forward;
        } else {
            update[i]->level[i].span -= 1;
        }
    }
    if (x->level[0].forward) {
        x->level[0].forward->backward = x->backward;
    } else {
        zsl->tail = x->backward;
    }
    while (zsl->level > 1 && zsl->header->level[zsl->level - 1].forward == NULL)
        zsl->level--;
    zsl->length--;
}

/* Delete an element with matching score/element from the skiplist.
 * The function returns 1 if the node was found and deleted, otherwise
 * 0 is returned. */
static int zslDelete(zskiplist *zsl, double score, const std::string &ele, zskiplistNode **node) {
    zskiplistNode *update[ZSKIPLIST_MAXLEVEL], *x;
    int i;

    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        while (x->level[i].forward &&
               (x->level[i].forward->score < score ||
                (x->level[i].forward->score == score &&
                 x->level[i].forward->ele < ele))) {
            x = x->level[i].forward;
        }
        update[i] = x;
    }
    /* We may have multiple elements with the same score, what we need
     * is to find the element with both the right score and object. */
    x = x->level[0].forward;
    if (x && score == x->score && x->ele == ele) {
        zslDeleteNode(zsl, x, update);
        if (!node)
            zslFreeNode(x);
        else
            *node = x;
        return 1;
    }
    return 0; /* not found */
}

/* Update the score of an element inside the sorted set skiplist.
 * Note that the element must exist and must match 'score'. */
static zskiplistNode *zslUpdateScore(zskiplist *zsl, double curscore, const std::string &ele, double newscore) {
    zskiplistNode *update[ZSKIPLIST_MAXLEVEL], *x;
    int i;

    /* We need to seek to element to update to start: this is useful anyway,
     * we'll have to update or remove it. */
    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        while (x->level[i].forward &&
               (x->level[i].forward->score < curscore ||
                (x->level[i].forward->score == curscore &&
                 x->level[i].forward->ele < ele))) {
            x = x->level[i].forward;
        }
        update[i] = x;
    }

    /* Jump to our element: note that this function assumes that the
     * element with the matching score exists. */
    x = x->level[0].forward;
    assert(x && curscore == x->score && x->ele == ele);

    /* If the node, after the score update, would be still exactly
     * at the same position, we can just update the score without
     * actually removing and re-inserting the element in the skiplist. */
    if ((x->backward == NULL || x->backward->score < newscore) &&
        (x->level[0].forward == NULL || x->level[0].forward->score > newscore)) {
        x->score = newscore;
        return x;
    }

    /* No way to reuse the old node: we need to remove and insert a new
     * one at a different place. */
    zslDeleteNode(zsl, x, update);
    zskiplistNode *newnode = zslInsert(zsl, newscore, x->ele);
    zslFreeNode(x);
    return newnode;
}

static int zslValueGteMin(double value, zrangespec *spec) {
    return spec->minex ? (value > spec->min) : (value >= spec->min);
}

static int zslValueLteMax(double value, zrangespec *spec) {
    return spec->maxex ? (value < spec->max) : (value <= spec->max);
}

/* Returns if there is a part of the zset is in range. */
static int zslIsInRange(zskiplist *zsl, zrangespec *range) {
    zskiplistNode *x;

    /* Test for ranges that will always be empty. */
    if (range->min > range->max ||
        (range->min == range->max && (range->minex || range->maxex)))
        return 0;
    x = zsl->tail;
    if (x == NULL || !zslValueGteMin(x->score, range))
        return 0;
    x = zsl->header->level[0].forward;
    if (x == NULL || !zslValueLteMax(x->score, range))
        return 0;
    return 1;
}

/* Find the first node contained in the specified range, NULL if none.
 * The range is inclusive on both ends. */
static zskiplistNode *zslFirstInRange(zskiplist *zsl, zrangespec *range) {
    zskiplistNode *x;
    int i;

    if (!zslIsInRange(zsl, range)) return NULL;

    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        /* Go forward while *OUT* of range. */
        while (x->level[i].forward && !zslValueGteMin(x->level[i].forward->score, range)) {
            x = x->level[i].forward;
        }
    }

    /* This is an inner range, so the next node cannot be NULL. */
    x = x->level[0].forward;
    assert(x != NULL);
    return x;
}

/* Find the rank for an element by both score and key.
 * Returns 0 when the element cannot be found, rank otherwise.
 * Note that the rank is 1-based due to the span of zsl->header to the
 * first element. */
static unsigned long zslGetRank(zskiplist *zsl, double score, const std::string &ele) {
    zskiplistNode *x;
    unsigned long rank = 0;
    int i;

    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        while (x->level[i].forward &&
               (x->level[i].forward->score < score ||
                (x->level[i].forward->score == score &&
                 x->level[i].forward->ele <= ele))) {
            rank += x->level[i].span;
            x = x->level[i].forward;
        }

        /* x might be equal to zsl->header, so test if obj is non-NULL */
        if (x->ele.size() && x->score == score && x->ele == ele) {
            return rank;
        }
    }
    return 0;
}

/* Finds an element by its rank. The rank argument needs to be 1-based. */
static zskiplistNode *zslGetElementByRank(zskiplist *zsl, unsigned long rank) {
    zskiplistNode *x;
    unsigned long traversed = 0;
    int i;

    x = zsl->header;
    for (i = zsl->level - 1; i >= 0; i--) {
        while (x->level[i].forward && (traversed + x->level[i].span) <= rank) {
            traversed += x->level[i].span;
            x = x->level[i].forward;
        }
        if (traversed == rank) {
            return x;
        }
    }
    return NULL;
}
