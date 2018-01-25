
document.addEventListener("keydown", function ( event ) {
    console.log(event.keyCode);
    if ( event.keyCode === 9 )  {
        event.stopPropagation();
        event.preventDefault();
    }
}, false);

impress().init();
