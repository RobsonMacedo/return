function olaMundo()
{
    alert('Testando o Js!')
}


window.onload = function(){
    const spinnerBox = document.getElementById('spinner-box') 
    const buttonSearch = document.getElementById('button-search')
    const containerIndex = document.getElementById('container-index')
    
    buttonSearch.onclick = function(){
    containerIndex.classList.add('not-visible')
    spinnerBox.classList.remove('not-visible')
    $.ajax({
        type:'POST',
        url:'/resultados/',
        success:function(response){
            spinnerBox.classList.add('not-visible')
        },
        error:function(error){
            console.log(error)
        }
        })
    }
}

