setwd("/home/tan/Documents/work/phase_filed/")
# t[K], P[bar]
get_free_energy <- function(x, t, P) {
    R = 8.31446262
    w_a = 19550 - 10.5 * t + 0.327
    w_b = 22820 - 6.3 * t + 0.461 * P
    w = x * w_a + (1 - x) * w_b

    R * t * (
        x * log(x) + (1 - x) * log(1 - x)
    ) + w * x * (1 - x)
}

get_binode <- function(x, th=.35) {
    n = length(g)

    # baseline estimate from median
    get_baseline0 = function(g, bline) {
        dg = diff(g)
        # median i
        i = which.min(abs(g - median(g)))
        bline = bline + median(dg) * (1:n - i) + g[i]
        return (bline)
    }

    # divide upper and lower part and use smallest one in each fraction
    get_baseline1 = function(g) {
        i = which.min(g[1:floor(n*th)])
        j = which.min(g[floor(n*th):n])
        bline = (g[i] - g[j]) / (i - j) * (1:n - j) + g[j]
        return (bline)
    }

    # simplest: use index i and j
    get_baseline3 <- function(i, j, g) {
        bline = (g[i] - g[j]) / (i - j) * (1:n - j) + g[j]
        return (bline)
    }

    # get index of dg = 0
    get_dg_0 <- function(g) {
        dg = diff(g)
        is_dg_0 = c(diff(dg > 0) == 1,FALSE)
        return(which(is_dg_0))
    }

    bline = numeric(n)
    bline = get_baseline0(g, bline)
    bline = get_baseline1(g)
    if (length(get_dg_0(g))) {
        for (i in 1:3) {
            is_dg_0 <- get_dg_0(g - bline)
            k <- is_dg_0[1]
            l <- is_dg_0[2]
            bline = bline + get_baseline3(k, l, g - bline)
            if (length(is_dg_0) != 2) {
                warning("dg don't have three convex point")
                return (NULL)
            }
        }
    } else {
        for (i in 1:3) {
            is_dg_0 <- get_dg_0(g - bline)
            k <- is_dg_0[1]
            l <- is_dg_0[2]
            bline = bline + get_baseline3(k, l, g - bline)
            if (length(is_dg_0) != 2) {
                warning("dg don't have three convex point")
                return (NULL)
            }
        }
    }
    return (c(k,l))
}

get_spinode <- function(g) {
    ddg_is_0 = which(as.logical(c(FALSE, abs(diff(diff(diff(g)) > 0)), FALSE)))
    if (length(ddg_is_0) != 2) {
        return (NULL)
    }
    return(ddg_is_0)
}

c = seq(0.01, 0.99, length=2000)
tc <- data.frame(t=c(),sp_c=c(),bi_c=c())
t0 = 273.15

for (t in c(seq(100,1000,length=200),seq(900,1000,length=100))) {
    g <- get_free_energy(c, t, 1)
    bi <- get_binode(g)
    sp <- get_spinode(g)
    if (is.null(bi) || is.null(sp)) {
        next
    } else {
        tc = rbind(
            tc,
            data.frame(
                t = c(t-t0, t-t0),
                sp_c = c(c[bi[1]], c[bi[2]]),
                bi_c = c(c[sp[1]], c[sp[2]])
            )
        )
    }
}

tc <- tc[order(tc$sp_c),]
write.csv(tc, file = "tc_space/strain_free_spinode_binode.csv")

pdf(width=5, height=6, "tc_space/tc_space_compile.pdf")
plot(
    NA, NA,
    xlim=c(0,1),ylim=c(250,650),
    xlab = "K-mole fraction",
    ylab = expression(T~"[\u00B0C]"),
    xaxs="i",
    )
lines(tc$sp_c, tc$t)
lines(tc$bi_c, tc$t)
dev.off()
