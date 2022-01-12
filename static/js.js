function start() {
    // Buid query parameter
        var param = {};
        // バカなので手打ちします。バカなので
        param["symbol_0"] = document.getElementById("symbol_0").value;
        param["symbol_1"] = document.getElementById("symbol_1").value;
        param["symbol_2"] = document.getElementById("symbol_2").value;
        param["symbol_3"] = document.getElementById("symbol_3").value;
        param["symbol_4"] = document.getElementById("symbol_4").value;
        param["symbol_5"] = document.getElementById("symbol_5").value;
        param["symbol_6"] = document.getElementById("symbol_6").value;
        param["symbol_7"] = document.getElementById("symbol_7").value;
        param["symbol_8"] = document.getElementById("symbol_8").value;
        param["symbol_9"] = document.getElementById("symbol_9").value;
        var query = jQuery.param(param);

    // Query with a new parameter
        $.get("/run" + "?" + query, function(data){});
};

function update() {
    // Buid query parameter
        var param = {};
        // バカなので手打ちします。バカなので
        param["symbol_0"] = document.getElementById("symbol_0").value;
        param["symbol_1"] = document.getElementById("symbol_1").value;
        param["symbol_2"] = document.getElementById("symbol_2").value;
        param["symbol_3"] = document.getElementById("symbol_3").value;
        param["symbol_4"] = document.getElementById("symbol_4").value;
        param["symbol_5"] = document.getElementById("symbol_5").value;
        param["symbol_6"] = document.getElementById("symbol_6").value;
        param["symbol_7"] = document.getElementById("symbol_7").value;
        param["symbol_8"] = document.getElementById("symbol_8").value;
        param["symbol_9"] = document.getElementById("symbol_9").value;
        var query = jQuery.param(param);

    // Query with a new parameter
        $.get("/update" + "?" + query, function(data){});
};
    
    //
    // Register Event handler
    //
    document.getElementById("symbol_0").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_1").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_2").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_3").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_4").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_5").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_6").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_7").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_8").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("symbol_9").addEventListener("change", function(){
        update();
    }, false);
    document.getElementById("start").addEventListener("click", function(){
        start();
    }, false);