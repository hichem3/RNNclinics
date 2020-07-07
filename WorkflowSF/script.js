//fileloader in textarea
// t1 : c'est l'id du textarea cible
// effacez les cookies du navigateur aprés chaques changement de ce script
/*
var clear = function(){
document.getElementById("number").value ="0";
document.getElementById("t1,t2").val("");
}
*/

var openFile = function(event) {
	var reader = new FileReader();
	reader.readAsText((event.target).files[0]);
    reader.onload = function() {

		 $("#t1").val(reader.result);
		 $("#t1").trigger("change");
    }; 
};








  
    
    
    
    
    
   
    
    
    



