import bokeh
from bokeh.io import save
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Range1d, LabelSet, Label
)

from sklearn.decomposition import PCA

def saveplot(model, path):
    pca = PCA(n_components=2)
    XY = pca.fit_transform(model[model.wv.vocab])

    source = ColumnDataSource(
        data=dict(x=XY[:, 0], y=XY[:, 1], labels=list(model.wv.vocab.keys())))


    p = figure(plot_width=600, plot_height=600)
    p.scatter(x='x', y='y', size=8, source=source)
    labels = LabelSet(x='x', y='y', text='labels', level='glyph',
                    x_offset=5, y_offset=5, source=source, render_mode='canvas')

    p.add_layout(labels)
    save(p, filename=path)