var output = "C:/Users/Braedyn Au/Desktop/septin imgs/";

var h = getHeight();
var w = getWidth();
var title = getTitle();

var h1 = floor(h/4);
var w1 = floor(w/4);

function shifth(x){
	x2 = x + h1;
	return x2;
}
function shiftw(y) {
	y2 = y + w1;
	return y2;
}

x = 0;
y = 0;
for (i=0;i<4;i++){
	x = 0;
	h2 = h1 + 60;
	for (j=0;j<4;j++){
		selectWindow(title);
		w2 = w1 + 60;
		run("Specify...", "width=w2 height=h2 x=x y=y");
		run("Duplicate...", " ");
		name = getTitle();
		save(output+name);
		x = shifth(x);
	}
	y = shiftw(y);
}

run("Close All");

