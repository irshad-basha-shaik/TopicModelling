(function(send) {
    XMLHttpRequest.prototype.send = function(body) {
        try{
            body.append("query_auth", window.localStorage.auth_query);
        }
        catch(err){
            console.log("query auth not appended.");
        }
        send.call(this, body);
    }
})(XMLHttpRequest.prototype.send);
