#include "neuralnetviewprovider.h"

#include "backend.h"
#include "neuralnet.h"
#include <QDebug>

#include <QPainter>

namespace {

QColor colorOf(qreal v)
{
    QColor clr;
    if(v < 0)
        clr.setBlueF(abs(v));
    else
        clr.setRedF(v);

    return clr;
}

}

QImage NeuralNetViewProvider::requestNetView(NeuralNet *net)
{
#ifdef NEURALNET_DURATION_INFO
    static auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();
#endif // NEURALNET_DURATION_INFO

    const int spacing = 40; // 40

    int index = 0; //id.toInt();

    QImage img(600, 1000, QImage::Format_RGB32); // 300 x 500
    img.fill({0xee, 0xee, 0xee});

    QPen pen;

    QPainter painter;
    painter.begin(&img);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(pen);

    // we can calculate the position of a neuron: x: layer * spacing + spacing
    //                                            y: index in layer * spacing + spacing

    // make it efficient by drawint the connections from the layer to the previous layer, then the neurons.
    const auto neurons = net->neurons();
    const int layerCount = neurons.length();
    for(int lI = layerCount - 1; lI > -1; --lI) { // > -1 so we get also 0 as value(the input layer)
        const int neuronCount = neurons[lI].length();
        for(int nI = 0; nI < neuronCount; ++nI) {
            // draw the connections
            const auto n = neurons[lI][nI];
            for(auto c: n.connections) {
                painter.setPen(colorOf(c.weight));
                painter.drawLine(QLine(QPoint{lI * spacing + spacing, nI * spacing + spacing},
                                       QPoint{c.target.layer * spacing + spacing, c.target.neuron * spacing + spacing}));
            }

            // draw the neuron
            QPen p(Qt::darkGreen);
            p.setWidth(3);
            painter.setPen(p);
            painter.drawEllipse(QPoint{lI * spacing + spacing, nI * spacing + spacing}, 4, 4);
            p.setWidth(1);
            painter.setPen(p);
        }
    }


#ifdef NEURALNET_DURATION_INFO
    auto end = clock.now();
    std::chrono::nanoseconds _diff = (end - start);
    qreal diff = _diff.count() * 1. / (std::nano::den / std::milli::den); // nano -> ms

    qInfo().nospace() << Q_FUNC_INFO << ": drawing image took " << diff << "ms";
#endif // NEURALNET_DURATION_INFO

    return img;
}

QPixmap NeuralNetViewProvider::requestNetPixmap(NeuralNet *net)
{
    return QPixmap::fromImage(requestNetView(net));
}
