# CeltIA web demo

Abre el archivo HTML directamente en el navegador o sirvelo localmente con un servidor estático.

Ejemplo:

```powershell
cd D:\mini-council
.\.venv\Scripts\Activate.ps1
python -m http.server 4173 --directory apps/web
```

Luego abre http://localhost:4173

La interfaz envía peticiones al servicio local de FastAPI en http://localhost:18080/v1/chat/completions.
