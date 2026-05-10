function! s:DeleteAllComments() range
  " Deletes all tex comments (trailing and full line)
  " without changing paragraph formatting.
  " Put this function in $HOME/.vim/ftplugin/tex.vim
  let l:save = winsaveview() " Save cursor position
  " First: Remove full comment lines completely (to retain formatting)
  global/\m^\s*%.*$/delete
  " Second: Remove trailing comments (but dont match escaped \%)
  %smagic/[^\\]\zs%.*//eI
  " Third: Join multiple blank lines to one (aesthetics)
  %smagic/\(\n\s*\)\{3,}/\r\r/eI
  call winrestview(l:save) " Restore cursor position
endfunction
command! -buffer -nargs=0 DeleteAllComments :call <SID>DeleteAllComments()
