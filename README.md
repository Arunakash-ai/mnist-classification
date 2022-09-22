# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images the dataset contains 70000 samples which is generated from tensorflow

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict



## PROGRAM
```python3
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Arunakash-ai/mnist-classification/blob/main/deep03.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Ms2HU22Nmxkg"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "from tensorflow.keras.datasets import mnist\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "from tensorflow.keras import utils\n",
        "import pandas as pd\n",
        "from sklearn.metrics import classification_report,confusion_matrix\n",
        "from tensorflow.keras.preprocessing import image"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gyyDcEJBoPWh",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d392c73a-c922-458a-cecc-6b3efc493fc1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
            "11493376/11490434 [==============================] - 0s 0us/step\n",
            "11501568/11490434 [==============================] - 0s 0us/step\n"
          ]
        }
      ],
      "source": [
        "(X_train, y_train), (X_test, y_test) = mnist.load_data()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "W0thCGmwocfQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8a20438b-4717-4422-d403-583202599bd3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ],
      "source": [
        "X_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Kl1HVshDojow",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "58b04260-fc30-4303-fd32-3e717a3065b8"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ],
      "source": [
        "X_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sUtPtTH8pYho"
      },
      "outputs": [],
      "source": [
        "single_image= X_train[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d_7A8n_JpexA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d8bc21a0-8ebb-4bc5-8246-0a5afb2caa3c"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ],
      "source": [
        "single_image.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qyuxyqKZpiAY",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "615b0815-c251-43eb-95d6-0d6ae1b0bb93"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f95a88d6a10>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAN9klEQVR4nO3df4xV9ZnH8c+zWP6QojBrOhKKSyEGg8ZON4gbl6w1hvojGhw1TSexoZE4/YNJaLIhNewf1WwwZBU2SzTNTKMWNl1qEzUgaQouoOzGhDgiKo5LdQ2mTEaowZEf/mCHefaPezBTnfu9w7nn3nOZ5/1Kbu6957nnnicnfDi/7pmvubsATH5/VXYDAJqDsANBEHYgCMIOBEHYgSAuaubCzIxT/0CDubuNN72uLbuZ3Wpmh8zsPTN7sJ7vAtBYlvc6u5lNkfRHSUslHZH0qqQudx9IzMOWHWiwRmzZF0t6z93fd/czkn4raVkd3weggeoJ+2xJfxrz/kg27S+YWbeZ9ZtZfx3LAlCnhp+gc/c+SX0Su/FAmerZsg9KmjPm/bezaQBaUD1hf1XSlWb2HTObKulHkrYV0xaAouXejXf3ETPrkbRD0hRJT7n724V1BqBQuS+95VoYx+xAwzXkRzUALhyEHQiCsANBEHYgCMIOBEHYgSAIOxAEYQeCIOxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EARhB4Ig7EAQhB0IgrADQRB2IAjCDgRB2IEgCDsQBGEHgiDsQBCEHQiCsANBEHYgCMIOBJF7yGZcGKZMmZKsX3rppQ1dfk9PT9XaxRdfnJx3wYIFyfrKlSuT9ccee6xqraurKznv559/nqyvW7cuWX/44YeT9TLUFXYzOyzppKSzkkbcfVERTQEoXhFb9pvc/aMCvgdAA3HMDgRRb9hd0k4ze83Musf7gJl1m1m/mfXXuSwAdah3N36Juw+a2bckvWhm/+Pue8d+wN37JPVJkpl5ncsDkFNdW3Z3H8yej0l6XtLiIpoCULzcYTezaWY2/dxrST+QdLCoxgAUq57d+HZJz5vZue/5D3f/QyFdTTJXXHFFsj516tRk/YYbbkjWlyxZUrU2Y8aM5Lz33HNPsl6mI0eOJOsbN25M1js7O6vWTp48mZz3jTfeSNZffvnlZL0V5Q67u78v6bsF9gKggbj0BgRB2IEgCDsQBGEHgiDsQBDm3rwftU3WX9B1dHQk67t3707WG32baasaHR1N1u+///5k/dSpU7mXPTQ0lKx//PHHyfqhQ4dyL7vR3N3Gm86WHQiCsANBEHYgCMIOBEHYgSAIOxAEYQeC4Dp7Adra2pL1ffv2Jevz5s0rsp1C1ep9eHg4Wb/pppuq1s6cOZOcN+rvD+rFdXYgOMIOBEHYgSAIOxAEYQeCIOxAEIQdCIIhmwtw/PjxZH316tXJ+h133JGsv/7668l6rT+pnHLgwIFkfenSpcn66dOnk/Wrr766am3VqlXJeVEstuxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EAT3s7eASy65JFmvNbxwb29v1dqKFSuS8953333J+pYtW5J1tJ7c97Ob2VNmdszMDo6Z1mZmL5rZu9nzzCKbBVC8iezG/1rSrV+Z9qCkXe5+paRd2XsALaxm2N19r6Sv/h50maRN2etNku4quC8ABcv72/h2dz83WNaHktqrfdDMuiV151wOgILUfSOMu3vqxJu790nqkzhBB5Qp76W3o2Y2S5Ky52PFtQSgEfKGfZuk5dnr5ZK2FtMOgEapuRtvZlskfV/SZWZ2RNIvJK2T9DszWyHpA0k/bGSTk92JEyfqmv+TTz7JPe8DDzyQrD/zzDPJeq0x1tE6aobd3buqlG4uuBcADcTPZYEgCDsQBGEHgiDsQBCEHQiCW1wngWnTplWtvfDCC8l5b7zxxmT9tttuS9Z37tyZrKP5GLIZCI6wA0EQdiAIwg4EQdiBIAg7EARhB4LgOvskN3/+/GR9//79yfrw8HCyvmfPnmS9v7+/au2JJ55IztvMf5uTCdfZgeAIOxAEYQeCIOxAEIQdCIKwA0EQdiAIrrMH19nZmaw//fTTyfr06dNzL3vNmjXJ+ubNm5P1oaGhZD0qrrMDwRF2IAjCDgRB2IEgCDsQBGEHgiDsQBBcZ0fSNddck6xv2LAhWb/55vyD/fb29ibra9euTdYHBwdzL/tClvs6u5k9ZWbHzOzgmGkPmdmgmR3IHrcX2SyA4k1kN/7Xkm4dZ/q/untH9vh9sW0BKFrNsLv7XknHm9ALgAaq5wRdj5m9me3mz6z2ITPrNrN+M6v+x8gANFzesP9S0nxJHZKGJK2v9kF373P3Re6+KOeyABQgV9jd/ai7n3X3UUm/krS42LYAFC1X2M1s1pi3nZIOVvssgNZQ8zq7mW2R9H1Jl0k6KukX2fsOSS7psKSfunvNm4u5zj75zJgxI1m/8847q9Zq3StvNu7l4i/t3r07WV+6dGmyPllVu85+0QRm7Bpn8pN1dwSgqfi5LBAEYQeCIOxAEIQdCIKwA0FwiytK88UXXyTrF12Uvlg0MjKSrN9yyy1Vay+99FJy3gsZf0oaCI6wA0EQdiAIwg4EQdiBIAg7EARhB4KoedcbYrv22muT9XvvvTdZv+6666rWal1Hr2VgYCBZ37t3b13fP9mwZQeCIOxAEIQdCIKwA0EQdiAIwg4EQdiBILjOPsktWLAgWe/p6UnW77777mT98ssvP++eJurs2bPJ+tBQ+q+Xj46OFtnOBY8tOxAEYQeCIOxAEIQdCIKwA0EQdiAIwg4EwXX2C0Cta9ldXeMNtFtR6zr63Llz87RUiP7+/mR97dq1yfq2bduKbGfSq7llN7M5ZrbHzAbM7G0zW5VNbzOzF83s3ex5ZuPbBZDXRHbjRyT9o7svlPR3klaa2UJJD0ra5e5XStqVvQfQomqG3d2H3H1/9vqkpHckzZa0TNKm7GObJN3VqCYB1O+8jtnNbK6k70naJ6nd3c/9OPlDSe1V5umW1J2/RQBFmPDZeDP7pqRnJf3M3U+MrXlldMhxB2109z53X+Tui+rqFEBdJhR2M/uGKkH/jbs/l00+amazsvosScca0yKAItTcjTczk/SkpHfcfcOY0jZJyyWty563NqTDSaC9fdwjnC8tXLgwWX/88ceT9auuuuq8eyrKvn37kvVHH320am3r1vQ/GW5RLdZEjtn/XtKPJb1lZgeyaWtUCfnvzGyFpA8k/bAxLQIoQs2wu/t/Sxp3cHdJNxfbDoBG4eeyQBCEHQiCsANBEHYgCMIOBMEtrhPU1tZWtdbb25uct6OjI1mfN29erp6K8MorryTr69evT9Z37NiRrH/22Wfn3RMagy07EARhB4Ig7EAQhB0IgrADQRB2IAjCDgQR5jr79ddfn6yvXr06WV+8eHHV2uzZs3P1VJRPP/20am3jxo3JeR955JFk/fTp07l6Quthyw4EQdiBIAg7EARhB4Ig7EAQhB0IgrADQYS5zt7Z2VlXvR4DAwPJ+vbt25P1kZGRZD11z/nw8HByXsTBlh0IgrADQRB2IAjCDgRB2IEgCDsQBGEHgjB3T3/AbI6kzZLaJbmkPnf/NzN7SNIDkv6cfXSNu/++xnelFwagbu4+7qjLEwn7LEmz3H2/mU2X9Jqku1QZj/2Uuz820SYIO9B41cI+kfHZhyQNZa9Pmtk7ksr90ywAztt5HbOb2VxJ35O0L5vUY2ZvmtlTZjazyjzdZtZvZv11dQqgLjV347/8oNk3Jb0saa27P2dm7ZI+UuU4/p9V2dW/v8Z3sBsPNFjuY3ZJMrNvSNouaYe7bxinPlfSdne/psb3EHagwaqFveZuvJmZpCclvTM26NmJu3M6JR2st0kAjTORs/FLJP2XpLckjWaT10jqktShym78YUk/zU7mpb6LLTvQYHXtxheFsAONl3s3HsDkQNiBIAg7EARhB4Ig7EAQhB0IgrADQRB2IAjCDgRB2IEgCDsQBGEHgiDsQBCEHQii2UM2fyTpgzHvL8umtaJW7a1V+5LoLa8ie/ubaoWm3s/+tYWb9bv7otIaSGjV3lq1L4ne8mpWb+zGA0EQdiCIssPeV/LyU1q1t1btS6K3vJrSW6nH7ACap+wtO4AmIexAEKWE3cxuNbNDZvaemT1YRg/VmNlhM3vLzA6UPT5dNobeMTM7OGZam5m9aGbvZs/jjrFXUm8Pmdlgtu4OmNntJfU2x8z2mNmAmb1tZquy6aWuu0RfTVlvTT9mN7Mpkv4oaamkI5JeldTl7gNNbaQKMzssaZG7l/4DDDP7B0mnJG0+N7SWmf2LpOPuvi77j3Kmu/+8RXp7SOc5jHeDeqs2zPhPVOK6K3L48zzK2LIvlvSeu7/v7mck/VbSshL6aHnuvlfS8a9MXiZpU/Z6kyr/WJquSm8twd2H3H1/9vqkpHPDjJe67hJ9NUUZYZ8t6U9j3h9Ra4337pJ2mtlrZtZddjPjaB8zzNaHktrLbGYcNYfxbqavDDPeMusuz/Dn9eIE3dctcfe/lXSbpJXZ7mpL8soxWCtdO/2lpPmqjAE4JGl9mc1kw4w/K+ln7n5ibK3MdTdOX01Zb2WEfVDSnDHvv51NawnuPpg9H5P0vCqHHa3k6LkRdLPnYyX38yV3P+ruZ919VNKvVOK6y4YZf1bSb9z9uWxy6etuvL6atd7KCPurkq40s++Y2VRJP5K0rYQ+vsbMpmUnTmRm0yT9QK03FPU2Scuz18slbS2xl7/QKsN4VxtmXCWvu9KHP3f3pj8k3a7KGfn/lfRPZfRQpa95kt7IHm+X3ZukLars1v2fKuc2Vkj6a0m7JL0r6T8ltbVQb/+uytDeb6oSrFkl9bZElV30NyUdyB63l73uEn01Zb3xc1kgCE7QAUEQdiAIwg4EQdiBIAg7EARhB4Ig7EAQ/w8ie3GmjcGk5QAAAABJRU5ErkJggg==\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "plt.imshow(single_image,cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gamIl8scp_vg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d05d3534-d6b0-4928-a762-97275a24c6fd"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000,)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ],
      "source": [
        "y_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a1cYOQDZHa1e",
        "outputId": "9d446ada-14d1-4a7e-eedb-1b836f425733"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000,)"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "p1Hr1eHcr7EB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cd7f72cf-15ac-45d7-d93d-a477e0a7cb82"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ],
      "source": [
        "X_train.min()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TbytbmcjsFcJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1b86e2e0-5a51-4636-cf34-7f0194b022de"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "255"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ],
      "source": [
        "X_train.max()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "D-L5mmALsIHR"
      },
      "outputs": [],
      "source": [
        "X_train_scaled = X_train/255.0\n",
        "X_test_scaled = X_test/255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O_5QWtIVsZZp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "42005798-8297-4dcc-f75d-1b788024c496"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.0"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ],
      "source": [
        "X_train_scaled.min()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RSjbbOiYse95",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "06ba4b04-8134-423e-d06d-d3a8c6ba3351"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "X_train_scaled.max()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DBXrOqnVqGTY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "71d6620f-4e3f-4298-abf2-8f5dedc7285d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "5"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ],
      "source": [
        "y_train[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oL7Pld1Qrd5x"
      },
      "outputs": [],
      "source": [
        "y_train_onehot = utils.to_categorical(y_train,10)\n",
        "y_test_onehot = utils.to_categorical(y_test,10)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZN9h128GrH_5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f864447c-fe60-4919-f266-7c775eed3093"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "numpy.ndarray"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ],
      "source": [
        "type(y_train_onehot)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BTaP6Ynlrp9p",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8f82ca3b-a309-49a1-91a3-873f79118e28"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 10)"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ],
      "source": [
        "y_train_onehot.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KJVyMJOSQpQi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "99262b7c-f5f2-4aa1-e63b-35bcf5196078"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f95a83c3a10>"
            ]
          },
          "metadata": {},
          "execution_count": 20
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "single_image = X_train[500]\n",
        "plt.imshow(single_image,cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ppoll2_iQY57",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8ff5aa34-282c-407e-d975-d4e0860795a0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ],
      "source": [
        "y_train_onehot[500]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-6H82O2ouNRq"
      },
      "outputs": [],
      "source": [
        "X_train_scaled = X_train_scaled.reshape(-1,28,28,1)\n",
        "X_test_scaled = X_test_scaled.reshape(-1,28,28,1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cXIbBlbasjaZ"
      },
      "outputs": [],
      "source": [
        "model = keras.Sequential()\n",
        "# Write your code here\n",
        "model.add(layers.Input(shape=(28,28,1)))\n",
        "model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))\n",
        "model.add(layers.MaxPool2D(pool_size=(2,2)))\n",
        "model.add(layers.Flatten())\n",
        "model.add(layers.Dense(32,activation='relu'))\n",
        "model.add(layers.Dense(10,activation='softmax'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H5g5Ek6CgssX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c1b972e4-8051-4e06-d7ee-eb90dd533d75"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 26, 26, 32)        320       \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         \n",
            " )                                                               \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 5408)              0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 32)                173088    \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 10)                330       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 173,738\n",
            "Trainable params: 173,738\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tx9Sw_xqHtqI"
      },
      "outputs": [],
      "source": [
        "# Choose the appropriate parameters\n",
        "model.compile(loss='categorical_crossentropy',\n",
        "              optimizer='adam',\n",
        "              metrics='accuracy')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oO6tpvb5Ii14",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "521f226b-4d3e-470e-90e2-b8d7315dce1f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "938/938 [==============================] - 27s 29ms/step - loss: 0.0279 - accuracy: 0.9916 - val_loss: 0.0480 - val_accuracy: 0.9847\n",
            "Epoch 2/5\n",
            "938/938 [==============================] - 27s 29ms/step - loss: 0.0225 - accuracy: 0.9929 - val_loss: 0.0456 - val_accuracy: 0.9857\n",
            "Epoch 3/5\n",
            "938/938 [==============================] - 25s 27ms/step - loss: 0.0190 - accuracy: 0.9942 - val_loss: 0.0484 - val_accuracy: 0.9858\n",
            "Epoch 4/5\n",
            "938/938 [==============================] - 26s 28ms/step - loss: 0.0149 - accuracy: 0.9955 - val_loss: 0.0485 - val_accuracy: 0.9850\n",
            "Epoch 5/5\n",
            "938/938 [==============================] - 26s 27ms/step - loss: 0.0127 - accuracy: 0.9958 - val_loss: 0.0620 - val_accuracy: 0.9826\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f959fd5e7d0>"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ],
      "source": [
        "model.fit(X_train_scaled ,y_train_onehot, epochs=5,\n",
        "          batch_size=64, \n",
        "          validation_data=(X_test_scaled,y_test_onehot))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "baRgwlwaLCqp"
      },
      "outputs": [],
      "source": [
        "metrics = pd.DataFrame(model.history.history)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yBCYG9r9LKsp",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "7fa55919-ad49-4729-a494-6a155a7caf53"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       loss  accuracy  val_loss  val_accuracy\n",
              "0  0.027852  0.991550  0.047988        0.9847\n",
              "1  0.022485  0.992917  0.045617        0.9857\n",
              "2  0.019028  0.994167  0.048428        0.9858\n",
              "3  0.014890  0.995467  0.048476        0.9850\n",
              "4  0.012675  0.995800  0.062010        0.9826"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2a2bfe9e-2113-4b4a-8699-2b07b3fcdc0b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>loss</th>\n",
              "      <th>accuracy</th>\n",
              "      <th>val_loss</th>\n",
              "      <th>val_accuracy</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.027852</td>\n",
              "      <td>0.991550</td>\n",
              "      <td>0.047988</td>\n",
              "      <td>0.9847</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.022485</td>\n",
              "      <td>0.992917</td>\n",
              "      <td>0.045617</td>\n",
              "      <td>0.9857</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.019028</td>\n",
              "      <td>0.994167</td>\n",
              "      <td>0.048428</td>\n",
              "      <td>0.9858</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.014890</td>\n",
              "      <td>0.995467</td>\n",
              "      <td>0.048476</td>\n",
              "      <td>0.9850</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.012675</td>\n",
              "      <td>0.995800</td>\n",
              "      <td>0.062010</td>\n",
              "      <td>0.9826</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2a2bfe9e-2113-4b4a-8699-2b07b3fcdc0b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2a2bfe9e-2113-4b4a-8699-2b07b3fcdc0b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2a2bfe9e-2113-4b4a-8699-2b07b3fcdc0b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ],
      "source": [
        "metrics.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4Sg3ECV6LMf5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "e5b18648-568b-4952-cbf7-670f21354a36"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f95a2c494d0>"
            ]
          },
          "metadata": {},
          "execution_count": 30
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "metrics[['accuracy','val_accuracy']].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A906k0lmLOgg",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "91ecdfdb-b824-419a-9730-b36506eacc6a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f959fd5e3d0>"
            ]
          },
          "metadata": {},
          "execution_count": 31
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "metrics[['loss','val_loss']].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PDnRigNeLk7B"
      },
      "outputs": [],
      "source": [
        "x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gP5Ud8DbLpvI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e3077951-aa49-4d77-be1d-4a11435d1ada"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 977    0    1    1    0    1    0    0    0    0]\n",
            " [   0 1132    3    0    0    0    0    0    0    0]\n",
            " [   1    1 1020    0    1    0    1    7    1    0]\n",
            " [   0    0    5 1001    0    3    0    0    1    0]\n",
            " [   0    0    0    0  977    0    0    0    0    5]\n",
            " [   0    0    1    8    0  882    1    0    0    0]\n",
            " [  12    3    1    1    4    5  931    0    1    0]\n",
            " [   0    1   11    2    0    0    0 1010    2    2]\n",
            " [   5    0   18    9    4    4    1    2  927    4]\n",
            " [   3    2    1   11   10    8    0    4    1  969]]\n"
          ]
        }
      ],
      "source": [
        "print(confusion_matrix(y_test,x_test_predictions))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9gJ7WV95L7my",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1aee67a1-8c9f-40eb-e2dc-05194b42dc18"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.98      1.00      0.99       980\n",
            "           1       0.99      1.00      1.00      1135\n",
            "           2       0.96      0.99      0.97      1032\n",
            "           3       0.97      0.99      0.98      1010\n",
            "           4       0.98      0.99      0.99       982\n",
            "           5       0.98      0.99      0.98       892\n",
            "           6       1.00      0.97      0.98       958\n",
            "           7       0.99      0.98      0.98      1028\n",
            "           8       0.99      0.95      0.97       974\n",
            "           9       0.99      0.96      0.97      1009\n",
            "\n",
            "    accuracy                           0.98     10000\n",
            "   macro avg       0.98      0.98      0.98     10000\n",
            "weighted avg       0.98      0.98      0.98     10000\n",
            "\n"
          ]
        }
      ],
      "source": [
        "print(classification_report(y_test,x_test_predictions))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KlBK9Iw_MHc0"
      },
      "source": [
        "**Prediction for a single input**\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9NlIpMcgPQS5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4fea09c8-9def-4739-f5d3-652694379521"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PIL.Image.Image"
            ]
          },
          "metadata": {},
          "execution_count": 77
        }
      ],
      "source": [
        "type(img)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Gho9nRGPMOO9"
      },
      "outputs": [],
      "source": [
        "img = image.load_img('gi.jpeg')\n",
        "img_tensor = tf.convert_to_tensor(np.asarray(img))\n",
        "img_28 = tf.image.resize(img_tensor,(28,28))\n",
        "img_28_gray = tf.image.rgb_to_grayscale(img_28)\n",
        "img_28_gray_scaled = img_28_gray.numpy()/255.0\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yrw9d6T8OXLh"
      },
      "outputs": [],
      "source": [
        "x_single_prediction = np.argmax(\n",
        "    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),\n",
        "     axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "J5YWILZSPgnJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "81fb1757-983d-46a7-92d8-2fdeaabe0ae7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[8]\n"
          ]
        }
      ],
      "source": [
        "print(x_single_prediction)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "P0De-3CVPpXZ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "0e7b4923-3aca-4ccd-b78f-29e32e015c3e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f95a06ae310>"
            ]
          },
          "metadata": {},
          "execution_count": 56
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qqh74INOfnjX"
      },
      "outputs": [],
      "source": [
        "img_28_gray_inverted = 255.0-img_28_gray\n",
        "img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.imshow(img_28_gray_inverted.numpy().reshape(28,28),cmap='gray')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "WlVPz63yQFzt",
        "outputId": "8f4b6fb4-514f-4232-f986-77672c0b8651"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f959feb5ad0>"
            ]
          },
          "metadata": {},
          "execution_count": 58
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "08peSjZ2f6xG"
      },
      "outputs": [],
      "source": [
        "x_single_prediction = np.argmax(\n",
        "    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),\n",
        "     axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jqoeXU7kf9Km",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "64fa5100-ef17-48b6-c8d4-1f651dc0c92e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[8]\n"
          ]
        }
      ],
      "source": [
        "print(x_single_prediction)"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "e4xop2HFN2Yf"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
