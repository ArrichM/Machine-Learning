library("tikzDevice")

pdf(file="summary_box.pdf", height = 13, width=9)
featurePlot(x = train_data[,-19], 
            y = train_data$default_time, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
dev.off()

pdf(file="summary_dens.pdf", height = 13, width=9)
featurePlot(x = train_data[,-19], 
            y = train_data$default_time, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
dev.off()


tikz(file="ROC.tex", height=5, width=5)
ROC_plot(caret_fit)
dev.off()

tikz(file="ROC.tex")
hist_plot()
dev.off()

pdf(file="ROC.pdf", height=8, width=8)
ROC_plot(caret_fit)
dev.off()

pdf(file="histograms.pdf", height = 11, width=8)
hist_plot()
dev.off()

pdf(file="histograms.pdf", height = 9, width=9)
hist_plot()
dev.off()

pdf(file="densities.pdf", height = 9, width=9)
densityplots_all()
dev.off()

pdf(file="resample_plot.pdf",  height = 9, width=9)
resample_plot(rValues)
dev.off()

pdf(file="resample_plot_dif.pdf",  height = 9, width=9)
resample_plot(difValues)
dev.off()


tikz(file = "skimmed.tex")
#
skimmed
dev.off()
