# Exact coefficient prescaling and streaming state

For each output h,g,j, substituting Vhat[t,g,j]=S[t,g](C[t,g,j]-8) into sum_t P[h,t]Vhat[t,g,j] gives sum_t W[h,t,g]C[t,g,j]-8sum_t W[h,t,g], with W=P*S. This is the distributive law over real numbers. The recoded signed form replaces C-8 with the exactly equal signed decoded nibble, including -8; no requantization is involved.

The reference materializes W and logical codes, not scaled dense V. It is an algebraic reference, not a native packed microkernel. In online attention P can be the unnormalized nonnegative block weights exp(z-m). DirectV supplies the weighted numerator. A block is summarized by its maximum m, weight sum l and weighted numerator o. Merge two blocks by rescaling l and o by exp(m_block-max(m_a,m_b)), then add. No full probability matrix is required by that composition.

FP64 dyadic fixtures are checked against an independent Fraction oracle. Numerical tests do not prove distributivity in floating-point; production accumulation policy and hardware lowering remain open.
