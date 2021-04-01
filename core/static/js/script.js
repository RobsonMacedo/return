


window.onload = function(){
    let spinnerBox = document.getElementById('spinner-box') 
    let buttonSearch = document.getElementById('button-search')
    let containerIndex = document.getElementById('container-index')
    

    buttonSearch.addEventListener('click', function(){
        containerIndex.classList.add('not-visible')
        spinnerBox.classList.remove('not-visible')
    })    
    
    
} 

