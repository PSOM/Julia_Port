# Port of ran3.f90 (Numerical Recipes ran3, Knuth subtractive generator).
# The state is kept between calls (Fortran SAVE), so the sequence continues across calls
# exactly as in the Fortran model. idum is a Ref: a negative value (or the first call) reinitialises.
mutable struct Ran3State
    iff::Int
    inext::Int
    inextp::Int
    ma::Vector{Int}
end
const ran3_state = Ran3State(0, 0, 0, zeros(Int, 55))

function ran3(idum::Ref{Int})
    MBIG = 1000000000
    MSEED = 161803398
    MZ = 0
    FAC = 1e0 / MBIG
    st = ran3_state
    ma = st.ma
    if (idum[] < 0 || st.iff == 0)
        st.iff = 1
        mj = mod(MSEED - abs(idum[]), MBIG)
        ma[55] = mj
        mk = 1
        for i in 1:54
            ii = mod(21 * i, 55)
            ma[ii] = mk
            mk = mj - mk
            if (mk < MZ)
                mk += MBIG
            end
            mj = ma[ii]
        end
        for k in 1:4
            for i in 1:55
                ma[i] = ma[i] - ma[1+mod(i + 30, 55)]
                if (ma[i] < MZ)
                    ma[i] += MBIG
                end
            end
        end
        st.inext = 0
        st.inextp = 31
        idum[] = 1
    end
    st.inext += 1
    st.inext == 56 && (st.inext = 1)
    st.inextp += 1
    st.inextp == 56 && (st.inextp = 1)
    mj = ma[st.inext] - ma[st.inextp]
    if (mj < MZ)
        mj += MBIG
    end
    ma[st.inext] = mj
    return mj * FAC
end
