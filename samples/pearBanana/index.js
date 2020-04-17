// Funtions to control web front-end elements,
// process input imamge, send to server,
// and get labelled image and count response.

(function() {
    var takeBtn = document.querySelector("#takeBtn");
    var photoFrame = document.querySelector("#photoFrame");
    $("#loading").hide();

    // Click upload button to send image
    document.querySelector("#upBtn").addEventListener("click", function() {
        var canvas = photoFrame.querySelector("canvas");
        if (canvas == null) {
            alert("Please take a photo or select a file.");
            return;
        }
        sendFile(canvas);
    });

})();