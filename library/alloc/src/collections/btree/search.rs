use core::borrow::Borrow;
use core::cmp::{Ordering, min};

use super::node::{marker, ForceResult::*, Handle, NodeRef};

use SearchResult::*;

pub enum SearchResult<BorrowType, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<BorrowType, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<BorrowType, K, V, GoDownType>, marker::Edge>),
}

/// Looks up a given key in a (sub)tree headed by the given node, recursively.
/// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
/// returns a `GoDown` with the handle of the possible leaf edge where the key
/// belongs.
pub fn search_tree<BorrowType, K, V, Q: ?Sized>(
    mut node: NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    key: &Q,
) -> SearchResult<BorrowType, K, V, marker::LeafOrInternal, marker::Leaf>
where
    Q: Ord,
    K: Borrow<Q>,
{
    loop {
        match search_node(node, key) {
            Found(handle) => return Found(handle),
            GoDown(handle) => match handle.force() {
                Leaf(leaf) => return GoDown(leaf),
                Internal(internal) => {
                    node = internal.descend();
                    continue;
                }
            },
        }
    }
}

/// Looks up a given key in a given node, without recursion.
/// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
/// returns a `GoDown` with the handle of the edge where the key might be found.
/// If the node is a leaf, a `GoDown` edge is not an actual edge but a possible edge.
pub fn search_node<BorrowType, K, V, Type, Q: ?Sized>(
    node: NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> SearchResult<BorrowType, K, V, Type, Type>
where
    Q: Ord,
    K: Borrow<Q>,
{
    match search_linear(&node, key) {
        (idx, true) => Found(unsafe { Handle::new_kv(node, idx) }),
        (idx, false) => SearchResult::GoDown(unsafe { Handle::new_edge(node, idx) }),
    }
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses binary search for better performance than linear search.
#[allow(dead_code)]
fn search_bsearch<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();

    let mut lo = 0;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = unsafe { node.reborrow().key_at(mid) };
        let mask = -((key <= k.borrow()) as isize) as usize;
        lo = (mask & lo) | (!mask & (mid + 1));
        hi = (mask & mid) | (!mask & hi);
        /*match key.cmp(k.borrow()) {
            Ordering::Greater => lo = mid + 1,
            Ordering::Equal => return (mid, true),
            Ordering::Less => hi = mid,
        }*/
    }

    (lo, lo < len && key == unsafe { node.reborrow().key_at(lo) }.borrow())
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses a skipping algorithm for better performance than linear search.
#[allow(dead_code)]
fn search_skip<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();
    const STEP: usize = 3;
    let mut lo = 0;
    let mut hi = STEP - 1;
    // Scan once, skipping keys to save time.
    // Ideally, the STEP amount should be the square root of `len`.
    while hi < len {
        let k = unsafe { node.reborrow().key_at(hi) };
        match key.cmp(k.borrow()) {
            Ordering::Greater => {},
            Ordering::Equal => return (hi, true),
            Ordering::Less => break,
        }
        lo = hi + 1;
        hi += STEP;
    }

    hi = min(hi, len);

    // Scan the small range where the key should be.
    while lo < hi {
        let k = unsafe { node.reborrow().key_at(lo) };
        match key.cmp(k.borrow()) {
            Ordering::Greater => {},
            Ordering::Equal => return (lo, true),
            Ordering::Less => return (lo, false),
        }
        lo += 1;
    }

    (hi, false)
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses unrolled linear search.
#[allow(dead_code)]
fn search_unroll<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len() & (!1usize);
    for i in (0..len).step_by(2) {
        let k1 = unsafe { node.reborrow().key_at(i) };
        let k2 = unsafe { node.reborrow().key_at(i + 1) };
        match key.cmp(k1.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false),
        }
        match key.cmp(k2.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (i + 1, true),
            Ordering::Less => return (i + 1, false),
        }
    }
    if node.len() % 2 > 0 {
        let idx = node.len() - 1;
        let k1 = unsafe { node.reborrow().key_at(idx) };
        match key.cmp(k1.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (idx, true),
            Ordering::Less => return (idx, false),
        }
    }
    (len, false)
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses linear search without early return.
#[allow(dead_code)]
fn search_all<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();
    let mut res = len;
    for i in 0..len {
        let k = unsafe { node.reborrow().key_at(i) };
        res = min(res, if key <= k.borrow() { i } else { len });
    }
    if res < len {
        let k = unsafe { node.reborrow().key_at(res) };
        (res, key == k.borrow())
    } else {
        (res, false)
    }
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses one iteration of binary search.
#[allow(dead_code)]
fn search_bsiter<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();

    let mid = len / 2;
    let mid_k = unsafe { node.reborrow().key_at(mid) };
    let cmp = key < mid_k.borrow();
    let mask = -(cmp as isize) as usize;
    let lo = !mask & mid;
    let hi = (mask & mid) | (!mask & len);
    //let lo = if cmp { 0 } else { mid };
    //let hi = if cmp { mid } else { len };

    for i in lo..hi {
        let k = unsafe { node.reborrow().key_at(i) };
        match key.cmp(k.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false),
        }
    }

    (hi, false)

    /*match key.cmp(mid_k.borrow()) {
        Ordering::Greater => {
            for i in (mid + 1)..len {
                let k = unsafe { node.reborrow().key_at(i) };
                match key.cmp(k.borrow()) {
                    Ordering::Greater => {}
                    Ordering::Equal => return (i, true),
                    Ordering::Less => return (i, false),
                }
            }

            (len, false)
        },
        Ordering::Equal => (mid, true),
        Ordering::Less => {
            for i in 0..mid {
                let k = unsafe { node.reborrow().key_at(i) };
                match key.cmp(k.borrow()) {
                    Ordering::Greater => {}
                    Ordering::Equal => return (i, true),
                    Ordering::Less => return (i, false),
                }
            }

            (mid, false)
        },
    }*/
}

#[allow(dead_code)]
fn search_monobound<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    if node.len() == 0 {
        return (0, false);
    }
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let mut lo = 0;
    let mut hi = node.len();
    while hi > 1 {
        let mid = hi / 2;
        let k = unsafe { node.reborrow().key_at(lo + mid) };
        if key >= k.borrow() {
            lo += mid;
        }
        hi -= mid;
    }
    let k = unsafe { node.reborrow().key_at(lo) };
    match key.cmp(k.borrow()) {
        Ordering::Greater => return (lo + 1, false),
        Ordering::Equal => return (lo, true),
        Ordering::Less => return (lo, false),
    }
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
/// This uses linear search.
#[allow(dead_code)]
fn search_linear<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();
    for i in 0..len {
        let k = unsafe { node.reborrow().key_at(i) };
        match key.cmp(k.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false),
        }
    }
    (len, false)
}
