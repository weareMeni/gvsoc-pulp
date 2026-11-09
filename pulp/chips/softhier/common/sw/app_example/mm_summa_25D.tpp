
#pragma once
#include "gemm.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef SUMMA25D_TRACE
#define SUMMA25D_TRACE 0
#endif
#ifndef SUMMA25D_CHUNK
#define SUMMA25D_CHUNK 8192
#endif

namespace summa25d_c4 {

using Half = _Float16 __attribute__((__may_alias__));

template<class C> struct Sync {
    uint32_t base=0, retired=0;
    volatile uint32_t ready_a[C::LOAD_DEPTH]={};
    volatile uint32_t ready_b[C::LOAD_DEPTH]={};
    volatile uint32_t pulled_a[C::LOAD_DEPTH]={};
    volatile uint32_t pulled_b[C::LOAD_DEPTH]={};
    volatile uint32_t c_ready[C::OUTPUT_DEPTH]={};
    volatile uint32_t c_ack[C::OUTPUT_DEPTH]={};
};

struct Bases {
    uintptr_t A=0, B=0, C=0;
};

template<class C>
static Sync<C> state
    __attribute__((section(".l1")));

template<class C>
static ClusterPipeline<C> pipe __attribute__((section(".l1")));

static Bases bases __attribute__((section(".l1")));

alignas(16) static Half scratch[3][SUMMA25D_CHUNK] __attribute__((section(".l1")));

inline void fence() {
    asm volatile("fence rw,rw" ::: "memory");
}

inline void pause() {
    asm volatile("" ::: "memory");
}

inline uint32_t off(const void* p) {
    return uint32_t(uintptr_t(p)-uintptr_t(local(0)));
}

inline FlexPosition2D peer(FlexPosition2D p,unsigned layer) {
    p.x=(p.x%2)+2*(layer%2);
    p.y=(p.y%2)+2*(layer/2);
    return p;
}

template<class C>
inline volatile uint32_t* flag(
    FlexPosition2D p, Sync<C>& s,
    size_t field, size_t slot
) {
    return (volatile uint32_t*)remote_pos(
        p,off(&s)+field+slot*sizeof(uint32_t)
    );
}

enum Event : uint8_t {
    A_HBM=1, B_HBM, A_PEER, B_PEER,
    A_DONE, B_DONE, INPUT_READY,
    REDMULE_SUBMIT, PARTIAL_READY,
    REDUCE_BEGIN, REDUCE_DONE, C_STORE, C_DONE
};

#if SUMMA25D_TRACE

struct Record {
    uint32_t cycle,ticket,info;
    uint16_t event,slot;
};

template<class C> struct Trace {
    Record data[2][256];
    uint32_t n[2]={},dropped[2]={};
};

template<class C>
static Trace<C> trace __attribute__((section(".l1")));

template<class C>
inline void log(
    uint32_t cid,uint32_t core,Event event,
    uint32_t ticket,uint32_t slot,uint32_t info=0
) {
    if(cid) return;

    auto& t=trace<C>;
    unsigned i=t.n[core];

    if(i>=256) {
        ++t.dropped[core];
        return;
    }

    uint32_t cycle;
    asm volatile("csrr %0,mcycle":"=r"(cycle));

    t.data[core][i]={
        cycle,ticket,info,
        (uint16_t)event,(uint16_t)slot
    };
    t.n[core]=i+1;
}

#else

template<class C>
inline void log(
    uint32_t,uint32_t,Event,
    uint32_t,uint32_t,uint32_t=0
) {}

#endif

// DMA-stage three peer partial C tiles, then reduce locally.
template<class C>
void reduce(
    FlexPosition2D pos,
    ClusterPipeline<C>& p,
    Sync<C>& s,
    size_t out,uint32_t ticket
) {
    Half* dst=(Half*)p.C_tiles[out].ptr();

    FlexPosition2D src[3]={
        peer(pos,1),peer(pos,2),peer(pos,3)
    };

    const uint32_t src_off=off(dst);

    for(size_t begin=0; begin<C::M*C::N; begin+=SUMMA25D_CHUNK) {

        const size_t n=(C::M*C::N-begin<SUMMA25D_CHUNK) ? C::M*C::N-begin : SUMMA25D_CHUNK;
        uint32_t id[3];

        for(unsigned l=0;l<3;++l) {
            id[l]=bare_dma_start_2d(
                dma_addr(scratch[l]),
                remote_pos(
                    src[l],
                    src_off+begin*sizeof(Half)
                ),
                n*sizeof(Half),
                n*sizeof(Half),
                n*sizeof(Half),
                1
            );
        }

        for(unsigned l=0;l<3;++l)
            while(!bare_dma_done(id[l]))
                pause();

        fence();

        for(size_t i=0;i<n;++i) {
            Half v=dst[begin+i];

            v+=scratch[0][i];
            v+=scratch[1][i];
            v+=scratch[2][i];

            dst[begin+i]=v;
        }
    }

    fence();

    // Every source can now reuse its C output buffer.
    for(unsigned l=0;l<3;++l)
        *flag(src[l],s,offsetof(Sync<C>,c_ack),out)
            =ticket;
}

} // namespace summa25d_c4

template<class C>
void mm_summa25d_c4(
    typename C::data_type* A,
    typename C::data_type* B,
    typename C::data_type* C_out,
    size_t M,size_t N,size_t K
) {
    using namespace summa25d_c4;
    using T=typename C::data_type;

    static_assert(
        C::GRID_ROWS==4 &&
        C::LOAD_DEPTH>=2 &&
        C::OUTPUT_DEPTH>=1,
        "c=4 requires a 4x4 mesh and pipeline buffers"
    );
    static_assert(
        sizeof(T)==2 && sizeof(Half)==2,
        "c=4 reduction requires FP16"
    );
    static_assert(
        SUMMA25D_CHUNK>0,
        "Reduction chunk must be nonempty"
    );

    const uint32_t cid=flex_get_cluster_id();
    const uint32_t core=flex_get_core_id();

    const size_t mt=M/C::M;
    const size_t nt=N/C::N;
    const size_t kt=K/C::K;

    if(!mt||!nt||!kt||
       M%C::M||N%C::N||K%C::K||
       mt%2||nt%2||kt%4) {
        if(!cid&&!core)
            flex_print("SUMMA25D: invalid tile dimensions\n");
        return;
    }

    const size_t mg_count=mt/2;
    const size_t ng_count=nt/2;
    const size_t local_kt=kt/4;
    const size_t total=mg_count*ng_count;

    const auto pos=get_pos(cid);

    const unsigned layer=
        (pos.y/2)*2+pos.x/2;

    const size_t row=pos.y%2;
    const size_t col=pos.x%2;

    auto& p = pipe<C>;
    auto& s=state<C>;
    auto& ps=p.sync;

    if(core==0) {
        s=Sync<C>{};
        ps=PipelineSync<C>{};
        s.base=redmule_completed_jobs;

        if(cid==0)
            bases={
                (uintptr_t)A,
                (uintptr_t)B,
                (uintptr_t)C_out
            };

#if SUMMA25D_TRACE
        if(cid==0)
            trace<C> = Trace<C>{};
#endif
    }

    flex_global_barrier_polling();

    if(core==0) {

        // Core 0: RedMulE exclusively.
        size_t gen=0,in=0,ticket=0;

        for(size_t mg=0;mg<mg_count;++mg)
        for(size_t ng=0;ng<ng_count;++ng,++ticket) {

            const size_t out=
                ticket%C::OUTPUT_DEPTH;

            if(ticket>=C::OUTPUT_DEPTH) {
                const uint32_t need=
                    ticket-C::OUTPUT_DEPTH+1;

                while(__atomic_load_n(
                    &s.retired,__ATOMIC_ACQUIRE
                )<need)
                    pause();
            }

            for(size_t k=0;k<local_kt;++k) {
                ++gen;

                while(__atomic_load_n(
                    &ps.inputs_arrived[in].completed,
                    __ATOMIC_ACQUIRE
                )<gen)
                    pause();

                p.run(in,out,k==0);

                log<C>(
                    cid,core,REDMULE_SUBMIT,
                    gen,in,out
                );

                in=(in+1)%C::LOAD_DEPTH;
            }
        }

    } else if(core==1) {

        // Core 1: all iDMA work and cross-layer reduction.
        const volatile Bases* b=
            cid==0 ? &bases :
            (const volatile Bases*)remote_cid(
                0,off(&bases)
            );

        const T* srcA=(const T*)(uintptr_t)b->A;
        const T* srcB=(const T*)(uintptr_t)b->B;
        T* dstC=(T*)(uintptr_t)b->C;

        const uint32_t base=s.base;

        constexpr size_t RA=
            offsetof(Sync<C>,ready_a);
        constexpr size_t RB=
            offsetof(Sync<C>,ready_b);
        constexpr size_t PA=
            offsetof(Sync<C>,pulled_a);
        constexpr size_t PB=
            offsetof(Sync<C>,pulled_b);
        constexpr size_t CR=
            offsetof(Sync<C>,c_ready);

        size_t mg=0,ng=0,k=0,in=0,done=0;
        uint32_t gen=0;

        bool ap=false,bp=false;
        bool ad=false,bd=false;
        bool inputs_finished=false;
        bool partial=false,storing=false;

        while(!inputs_finished||done<total) {

            // Output: completion, reduction, and final store.
            if(done<total) {
                const uint32_t ticket=done+1;
                const size_t out=
                    done%C::OUTPUT_DEPTH;

                if(!partial &&
                   redmule_completed_jobs>=
                       base+ticket*local_kt) {

                    fence();
                    s.c_ready[out]=ticket;
                    fence();
                    partial=true;

                    log<C>(
                        cid,core,PARTIAL_READY,
                        ticket,out
                    );
                }

                if(partial&&layer!=0) {

                    if(s.c_ack[out]>=ticket) {
                        ++done;
                        partial=false;

                        __atomic_store_n(
                            &s.retired,
                            (uint32_t)done,
                            __ATOMIC_RELEASE
                        );
                    }

                } else if(partial&&layer==0) {

                    if(!storing) {
                        bool ready=true;

                        for(unsigned l=1;l<4;++l) {
                            if(*flag(
                                peer(pos,l),s,CR,out
                            )<ticket)
                                ready=false;
                        }

                        if(ready) {
                            log<C>(
                                cid,core,REDUCE_BEGIN,
                                ticket,out
                            );

                            reduce(
                                pos,p,s,out,ticket
                            );

                            log<C>(
                                cid,core,REDUCE_DONE,
                                ticket,out
                            );

                            const size_t m=
                                (done/ng_count)*2+row;
                            const size_t n=
                                (done%ng_count)*2+col;

                            p.submit_output(
                                dstC+m*C::M*N+n*C::N,
                                N,out
                            );

                            storing=true;

                            log<C>(
                                cid,core,C_STORE,
                                ticket,out,
                                ps.output_dma_id[out]
                            );
                        }

                    } else if(bare_dma_done(
                        ps.output_dma_id[out]
                    )) {
                        fence();

                        ++done;
                        partial=storing=false;

                        log<C>(
                            cid,core,C_DONE,
                            ticket,out,
                            ps.output_dma_id[out]
                        );

                        __atomic_store_n(
                            &s.retired,
                            (uint32_t)done,
                            __ATOMIC_RELEASE
                        );
                    }
                }
            }

            if(inputs_finished)
                continue;

            // Input: A/B ownership and 2x2 layer-local exchange.
            const size_t slot=in;
            const size_t km=layer*local_kt+k;

            const size_t m=mg*2+row;
            const size_t n=ng*2+col;

            const uint32_t next=gen+1;
            const unsigned owner=k&1u;

            const bool ownA=col==owner;
            const bool ownB=row==owner;

            auto a_src=pos,b_src=pos;

            a_src.x=(a_src.x/2)*2+owner;
            b_src.y=(b_src.y/2)*2+owner;

            const size_t old=__atomic_load_n(
                &ps.inputs_arrived[slot].completed,
                __ATOMIC_ACQUIRE
            );

            const bool compute_done=
                !old||
                redmule_completed_jobs>=base+old;

            const unsigned old_owner=
                old?(((old-1)%local_kt)&1u):0;

            // An owner cannot overwrite a tile until its peer
            // has completed the corresponding DMA transfer.
            const bool reuse_a=
                compute_done&&
                (!old||col!=old_owner||
                 s.pulled_a[slot]>=old);

            const bool reuse_b=
                compute_done&&
                (!old||row!=old_owner||
                 s.pulled_b[slot]>=old);

            if(!ap&&reuse_a) {

                if(ownA) {
                    ps.a_dma_id[slot]=bare_dma_start_2d(
                        dma_addr(p.A_tiles[slot].ptr()),
                        dma_addr(
                            srcA+m*C::M*K+km*C::K
                        ),
                        C::K*sizeof(T),
                        C::K*sizeof(T),
                        K*sizeof(T),
                        C::M
                    );

                    ap=true;

                    log<C>(
                        cid,core,A_HBM,
                        next,slot,ps.a_dma_id[slot]
                    );

                } else if(*flag(
                    a_src,s,RA,slot
                )>=next) {

                    const size_t bytes=
                        C::M*C::K*sizeof(T);

                    ps.a_dma_id[slot]=bare_dma_start_2d(
                        dma_addr(p.A_tiles[slot].ptr()),
                        remote_pos(
                            a_src,
                            off(p.A_tiles[slot].ptr())
                        ),
                        bytes,bytes,bytes,1
                    );

                    ap=true;

                    log<C>(
                        cid,core,A_PEER,
                        next,slot,ps.a_dma_id[slot]
                    );
                }
            }

            if(!bp&&reuse_b) {

                if(ownB) {
                    ps.b_dma_id[slot]=bare_dma_start_2d(
                        dma_addr(p.B_tiles[slot].ptr()),
                        dma_addr(
                            srcB+km*C::K*N+n*C::N
                        ),
                        C::N*sizeof(T),
                        C::N*sizeof(T),
                        N*sizeof(T),
                        C::K
                    );

                    bp=true;

                    log<C>(
                        cid,core,B_HBM,
                        next,slot,ps.b_dma_id[slot]
                    );

                } else if(*flag(
                    b_src,s,RB,slot
                )>=next) {

                    const size_t bytes=
                        C::K*C::N*sizeof(T);

                    ps.b_dma_id[slot]=bare_dma_start_2d(
                        dma_addr(p.B_tiles[slot].ptr()),
                        remote_pos(
                            b_src,
                            off(p.B_tiles[slot].ptr())
                        ),
                        bytes,bytes,bytes,1
                    );

                    bp=true;

                    log<C>(
                        cid,core,B_PEER,
                        next,slot,ps.b_dma_id[slot]
                    );
                }
            }

            if(ap&&!ad&&bare_dma_done(
                ps.a_dma_id[slot]
            )) {
                fence();

                if(!ownA)
                    *flag(a_src,s,PA,slot)=next;

                fence();
                s.ready_a[slot]=next;
                ad=true;

                log<C>(
                    cid,core,A_DONE,
                    next,slot,ps.a_dma_id[slot]
                );
            }

            if(bp&&!bd&&bare_dma_done(
                ps.b_dma_id[slot]
            )) {
                fence();

                if(!ownB)
                    *flag(b_src,s,PB,slot)=next;

                fence();
                s.ready_b[slot]=next;
                bd=true;

                log<C>(
                    cid,core,B_DONE,
                    next,slot,ps.b_dma_id[slot]
                );
            }

            if(!ad||!bd)
                continue;

            __atomic_store_n(
                &ps.inputs_arrived[slot].completed,
                (size_t)next,
                __ATOMIC_RELEASE
            );

            log<C>(
                cid,core,INPUT_READY,
                next,slot
            );

            gen=next;
            in=(in+1)%C::LOAD_DEPTH;

            ap=bp=ad=bd=false;

            if(++k==local_kt) {
                k=0;

                if(++ng==ng_count) {
                    ng=0;

                    if(++mg==mg_count)
                        inputs_finished=true;
                }
            }
        }
    }

    flex_global_barrier_polling();
}

template<class C>
void mm_summa25d_c4_dump_trace() {
#if SUMMA25D_TRACE
    using namespace summa25d_c4;

    if(flex_get_cluster_id()!=0||
       flex_get_core_id()!=0)
        return;

    auto& t=trace<C>;

    printf(
        "SUMMA25D,core,cycle,event,ticket,slot,info\n"
    );

    for(unsigned core=0;core<2;++core) {
        const unsigned n=
            t.n[core]<256?t.n[core]:256;

        for(unsigned i=0;i<n;++i) {
            const auto& e=t.data[core][i];

            printf(
                "SUMMA25D,%u,%u,%u,%u,%u,%u\n",
                core,e.cycle,
                (unsigned)e.event,
                e.ticket,e.slot,e.info
            );
        }

        printf(
            "SUMMA25D_DROPPED,%u,%u\n",
            core,t.dropped[core]
        );
    }
#endif
}
