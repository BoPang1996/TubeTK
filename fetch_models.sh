DIR='./models'
URL='https://drive.google.com/uc?id=1jLgyNmiZ_c-m8Cw3NcZTEPTf6VESfIzK&export=download'

mkdir -p $DIR

echo "Downloading pre-trained TubeTK..."
FILE="$(curl -sc /tmp/gcokie "${URL}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
curl -Lb /tmp/gcokie "${URL}&confirm=$(awk '/_warning_/ {print $NF}' /tmp/gcokie)" -o "$DIR/${FILE}"

echo "Download success."
