# blaidd

blaidd is a helper to distribute the colors of labels in a 2D scatterplot.

Let's say you want to make a 2-dimensional scatterplot and you'd like every datapoint to be colored according to a cluster label, but you have plenty of labels and so you risk to end up having nearby clusters with similar colors and not being able to distinguish them.

blaidd helps by trying to solve an optimization proble which maximizes the difference of scale between nearby clusters, trying to improve the visual effect of your scatterplot.