
echo "Starting experiment with PolypSegNet_Full model..."
python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_Full

echo "Starting experiment with PolypSegNet_Resnet_SCAS model..."
python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_Resnet_SCAS

echo "Starting experiment with PolypSegNet_Resnet_D2F model..."
python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_Resnet_D2F

echo "Starting experiment with PolypSegNet_Resnet model..."
python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_Resnet

echo "Starting experiment with PolypSegNet_D2F model..."
python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_D2F

 echo "Starting experiment with PolypSegNet_Baseline model..."
 python main.py --mode train --dataset cvc-clinicdb --model PolypSegNet_Baseline

echo "experiment completed."