function logout(){
    window.localStorage.removeItem('auth_query');
    var form = new FormData();
    var xhttp = new XMLHttpRequest();
    xhttp.open('POST', '/logout', true);
    xhttp.send();
    xhttp.onload = function () {
        if (xhttp.readyState === xhttp.DONE) {
            if(xhttp.status == 200){
             var resp = JSON.parse(this.responseText);
             console.log(resp);
             window.location = '/';
            }
        }
    };
}