template<typename C>
inline int ClusterPipeline<C>::run(
    size_t in_slot,
    size_t out_slot,
    bool first
) {
    uint8_t op = first ? MATMUL : GEMM;
    return submit_redmule(in_slot, out_slot, op);
}