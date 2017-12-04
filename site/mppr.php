<?
echo $_POST['data'];
for ($i=0; $i < count($_POST['data']); $i++) { 
    $text = $text.$_POST['data'][$i].',';
}
file_put_contents('test.txt',$text);

echo $text;
?>