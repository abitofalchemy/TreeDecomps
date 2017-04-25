# http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
# http://pseudofish.com/triangle-heatmaps-in-r-using-ggplot.html

# install.packages("corrplot")
library(corrplot)
filename='Results/ucidata-zachary_isom.tsv'
M<-cor(mtcars)
setwd("/Users/saguinag/Theory/Grammars/TreeDecomps")
# M<-matrix(read.csv('../Results/lesmis.dat'),nrow=6,byrow=TRUE)
M<- read.csv(filename, row.names=1)
M<-as.matrix(M)
head(round(M,2))

# corrplot(abs(M),order="AOE", cl.lim=c(0,1))

# ToDo: Export from python to files that look like .dat file for this example
pdf( file = sprintf("Results/%s.pdf",basename(filename)), width = 5, height = 4 )	# numbers are cm 
  corrplot(M, method="circle", 
         col=colorRampPalette(c("red","blue","white"))(100), cl.lim=c(0,1), type="lower")
  dev.off()