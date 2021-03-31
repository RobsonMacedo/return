


window.onload = function(){
    let spinnerBox = document.getElementById('spinner-box') 
    let buttonSearch = document.getElementById('button-search')
    let containerIndex = document.getElementById('container-index')
    

    buttonSearch.addEventListener('click', function(){
        containerIndex.classList.add('not-visible')
        spinnerBox.classList.remove('not-visible')
    })    
    
    
} 

/* testar isso novamente, jogar um script dentro do código direto pra testar */
/* let dataBox = document.getElementById('data-box')
dataBox.onload = function(e){
        console.log(e)
        Swal.fire('oi')
    }
 */
