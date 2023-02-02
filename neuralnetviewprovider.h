#ifndef NEURALNETVIEWPROVIDER_H
#define NEURALNETVIEWPROVIDER_H

#include <QImage>


class NeuralNet;

class NeuralNetViewProvider
{
public:
    static QImage requestNetView(NeuralNet *net);
    static QPixmap requestNetPixmap(NeuralNet *net);
};

#endif // NEURALNETVIEWPROVIDER_H
