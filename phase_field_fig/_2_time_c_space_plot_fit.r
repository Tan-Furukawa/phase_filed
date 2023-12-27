library(reticulate)
setwd("/home/tan/Documents/work/phase_filed/")
np <- import("numpy")
dat <- np$load("phase_field_fig/time_c_space.npy")

r <- 1:1500
dat <- dat[r,sample(1:ncol(dat))]
dtime = 0.004 * 100

png(
    "phase_field_fig/output.png",
    width=400,
    height=400,
    res = 100)
plot(
    NA,
    NA,
    xlim=c(-10,nrow(dat)*dtime),
    ylim=c(0,1),
    xlab = "time",
    ylab = "K-mole fraction",
    yaxs = "i",
    xaxs = "i"
    )

for (i in 1:nrow(dat)) {
    x = i + numeric(ncol(dat))
    points(
        (x * dtime),
        dat[i,],
        col=NULL,
        bg = rgb(0,0,0,alpha=.01),
        pch = 21,
        cex = .1
    )
}
text(max(x * dtime) * 0.75, 0.95, "T=700 K "~c[0]~"=0.4")
# legend(
#     "topright",
#     legend=paste("T=700 K ",expression(c),"=0.4",sep=""),
#     )
dev.off()

##==================================
# fig 2
##==================================

dat <- np$load("phase_field_fig/lambda_result.npy")
pdf("phase_field_fig/output2.pdf",
    width=4,
    height=4
    )
dat <- dat[r]
x <- (1:length(dat))*dtime
plot(
    NA,
    NA,
    xlim=c(-10,nrow(dat)*dtime),
    ylim=c(range(c(0,dat))),
    xlab = "time",
    ylab = "lamellae width",
    xaxs = "i"
    )
lines(x, dat)
text(max(x) * 0.25, max(dat)*0.95, "T=700 K "~c[0]~"=0.4")
dev.off()
