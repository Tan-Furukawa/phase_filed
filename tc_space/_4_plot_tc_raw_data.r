# making t-c space raw data plot
# making fitted_param.csv
setwd("/home/tan/Documents/work/phase_filed/")

library(magrittr)

strain_free <- read.csv("tc_space/strain_free_spinode_binode.csv")
d <- read.csv("tc_space/tc_space.csv")
d_growth = d[d$div1 < d$div2,]
d_growth <- d_growth[order(d_growth$t), ]
d_decline = d[d$div1 > d$div2,]
d_decline <- d_decline[rev(order(d_decline$t)), ]

res = data.frame(c0 =c(), transition_t=c())
for(c0 in unique(c(d_decline$c0, d_growth$c0))) {
    du = d_decline[d_decline$c0 == c0,]
    dd = d_growth[d_growth$c0 == c0,]
    res = rbind(
        res,
        data.frame(
            c0 = c0,
            transition_t = (min(du$t) + max(dd$t)) / 2
        )
    )
}

fit = lm(transition_t ~ I(c0^3)+I(c0^2)+c0+1, data=res)
write.csv(summary(fit)$coefficients,"tc_space/fitted_param.csv")
a0 = summary(fit)$coefficients[1,1]
a3 = summary(fit)$coefficients[2,1]
a2 = summary(fit)$coefficients[3,1]
a1 = summary(fit)$coefficients[4,1]

c = seq(0,1,length=100)
fitted <- (function(x) a3*x^3 + a2*x^2 + a1*x + a0)(c)

pdf(width=5, height=7, "tc_space/tc_space_raw.pdf")
plot(
    NA, NA,
    xlim=c(0,1),ylim=c(100,750),
    xlab = "K-mole fraction",
    ylab = expression(T~"[\u00B0C]"),
    xaxs="i",
    )
lines(c, fitted,
    lwd=2,
    col="grey20"
    )
lines(strain_free$sp_c, strain_free$t,
lwd=2,
col="grey50",
lty="dashed"
)
lines(strain_free$bi_c, strain_free$t,
lwd=2,
col="grey50",
lty="dashed"
)
points(
 d_growth$c0,
 d_growth$t,
 pch=24,
 bg="grey40",
 cex=.8
 )
points(
 d_decline$c0,
 d_decline$t,
 pch=25,
 bg="grey90",
 cex=.8
)
legend("topright",
legend=c("lamellar growth","single phase","coherent","strain free"),
    pch = c(24,25,NA,NA),
    pt.bg = c("grey40","grey90","",""),
    col= c("black","black","grey20","grey50"),
    lty = c(NA,NA,"solid", "dashed"),
    bg = "white"
)
dev.off()

max(fitted)
c[which.max(fitted)]
# 0.3636
# 508 C

max_strain_free = strain_free[which.max(strain_free$t),]
print("-------- delta T is ---------")
print(max_strain_free$t - max(fitted))
# delta_T = 132.6 C


