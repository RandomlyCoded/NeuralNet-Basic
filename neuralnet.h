#ifndef NEURALNET_H
#define NEURALNET_H

#include <QList>

#include "NeuralNet_global.h"

class QPoint;
class NeuralNet
{
public: // needs to be public so the user can use these types to create custom nets or modify single neurons. and the .cpp file needs it
    // neuron index
    struct Index {
        int layer; int neuron;
        Q_ALWAYS_INLINE bool operator==(Index other) { return layer == other.layer && neuron == other.neuron; }
        Q_ALWAYS_INLINE bool isValid() { return layer != -1 && neuron != -1; }
    };

    // a type that hold the data of the neuron
    struct Neuron {
        Neuron() {} /// constructor for empty neurons(e.g. input)

        // data type for each connection
        struct Connection {

            // the index of the neuron where the connection points to
            Index target;

            // target neuron value multiplier
            qreal weight;
        };

        Neuron(QList<Connection> connections, qreal bias, qreal activ) : v(0), connections(connections), bias(bias), activation(activ) {}
        Neuron(QList<Connection> connections) : Neuron(connections, 1, -1) {} // fire every time, no bias
        Neuron(Neuron const&other) : connections(other.connections), bias(other.bias), activation(other.activation) {}

        // all the connections of the neuron
        QList<Connection> connections;

        // my own value, used for calculating the following neurons
        qreal v;

        // divide the sum of the connections by this before normalizing it using the sigmoid function
        qreal bias;

        // activate only if the value after sigmoid is greater than this, otherwise set the value to 0
        qreal activation;
    };

public:
    /**
     * input: amount of input neurons
     * hiddenLayers: amount of hidden layers
     * hiddenLayerSize: amout of neurons per hidden layer
     * output: amout of output neurons **/
    explicit NeuralNet(int input, int hiddenLayers, int hiddenLayerSize, int output, int maxLayerWidth = 1000);
    NeuralNet(QList<QList<Neuron>> neurons) : m_neurons(neurons), m_maxLayerW(1000) {} // assuming the user inputs no nonsense

    void crossover(NeuralNet *partner);
    static NeuralNet *crossover(NeuralNet *a, NeuralNet *b);

    qsizetype size() { return m_neurons.size() + sizeof(m_maxLayerW); }

public slots:
    QList<qreal> decide(QList<qreal> in);

    void mutate(qreal mr);
    void addNeuron();
    void removeNeuron();

    const QList<QList<Neuron>> neurons() const { return m_neurons; }

private:
    Neuron neuronAt(Index idx);
    QList<QList<Neuron>> m_neurons;
    const int m_maxLayerW;

    void set(QList<QList<Neuron>> d);

    friend void writeNeuralNet(QDataStream &s, NeuralNet *net);
    friend void readNeuralNet(QDataStream&, NeuralNet *);
};

// converting index <-> point
QPoint toPoint(NeuralNet::Index idx);
NeuralNet::Index toIndex(QPoint p);

// reading and writing functions
void writeNeuralNet(QDataStream &s, NeuralNet *net);
QDataStream &operator<<(QDataStream &s, NeuralNet *net);

void readNeuralNet(QDataStream &s, NeuralNet *net);
QDataStream &operator>>(QDataStream &s, NeuralNet *net);


void writeNeuron(QDataStream &s, NeuralNet::Neuron neuron);
QDataStream &operator<<(QDataStream &s, NeuralNet::Neuron neuron);

void readNeuron(QDataStream &s, NeuralNet::Neuron &n);
QDataStream &operator>>(QDataStream &s, NeuralNet::Neuron &n);

#endif // NEURALNET_H
