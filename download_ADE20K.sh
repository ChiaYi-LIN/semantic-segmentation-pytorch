mkdir -p ../data
wget -O ../data/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ../data/ADEChallengeData2016.zip -d ../data
rm ../data/ADEChallengeData2016.zip
echo "ADEChallengeData2016 downloaded."