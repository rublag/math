function insum(t::BigFloat, m)
    k = 10
    a = sum( j -> ((t * ((2.)^(-(m+1))))^(2. *j) / (-2.)^j) / (factorial(big(2*j+k-1)) * factorial(big(j))) , (0:100))
    b = (t / (2.)^(m+1))^2 * sum( j -> ((t/((2.)^(m+1)))^(2. *j) / (-2.)^j) / (factorial(big(2*j+k+1)) * factorial(big(j))) , (0:100))
    a - b
end

function gm0(t, m)
    insum(convert(BigFloat, t), m) * factorial(9) / sqrt((2.)^(-m)) * 2 / sqrt(3*sqrt(pi))
end


function printUsage()
    println("Usage: julia gm0.jl m t")
end

function parseArgs(args)
    if length(args) < 2
        return nothing
    end

    m = tryparse(Int64, args[1])
    t = tryparse(Float64, args[2])

    if m == nothing || t == nothing
        return nothing
    end

    return (m, t)
end

function main(args)
    parsedArgs = parseArgs(args)
    if parsedArgs == nothing
        return printUsage()
    end
    m, t = parsedArgs
    println("Computing g_{$m,0} ($t).")

    res = convert(Float64, gm0(t, m))
    println("Result: g_{$m,0} ($t) = $res.")
end

using SpecialFunctions
const k = 10
setprecision(6000)
main(ARGS)
