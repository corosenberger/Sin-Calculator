import SimpleNet as sn
import math

tau = 2*math.pi

def compressInput(num): #compresses input so its between 0 and 1
    if num < 0:
        num *= -1
    print((num % tau))
    return [(num % tau) / tau]

def decompressOutput(num, negative=False): #decompresses output so it between -1 and 1
    num = num[0][0]
    if negative: #checks if the initial input was negative
        num *= -1
    return (num - .5) * 2

def main():
    layers, weights, biases = sn.readSave("sincalculator.txt") #retrieves previous training data
    net = sn.SimpleNet(layers,weights,biases)
    inputValue = 11
    print(decompressOutput(net.computeOutput(compressInput(inputValue)),inputValue < 0)) #prints the Neural Networks guess of sin(inputValu)
    print(math.sin(inputValue)) #prints the actual value of sin(inputValue) to test for accuracy
    
if __name__ == "__main__":
    main()