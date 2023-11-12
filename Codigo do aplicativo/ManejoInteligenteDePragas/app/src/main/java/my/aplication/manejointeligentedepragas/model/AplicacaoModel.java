package my.aplication.manejointeligentedepragas.model;

public class AplicacaoModel {
    private int Cod_Aplicacao;
    private String Data;
    private String DataPlano;
    private String MetodoAplicado;
    private int popPragas;
    private int numPlantas;
    private String Autor;
    private String Estacao;
    private String Temperatura;
    private String Tempo;
    private int fk_Cod_MetodoControle;
    private int fk_Cod_Talhao;
    private int fk_Cod_Praga;

    public AplicacaoModel(String Autor, int popPragas, int numPlantas, String DataPlano, String MetodoAplicado, int cod_Aplicacao, String data, String estacao, String temperatura, String tempo, int fk_Cod_MetodoControle, int fk_Cod_Praga, int fk_Cod_Talhao) {
        this.Cod_Aplicacao = cod_Aplicacao;
        this.Data = data;
        this.DataPlano = DataPlano;
        this.MetodoAplicado = MetodoAplicado;
        this.popPragas = popPragas;
        this.numPlantas = numPlantas;
        this.Estacao = estacao;
        this.Temperatura = temperatura;
        this.Tempo = tempo;
        this.fk_Cod_MetodoControle = fk_Cod_MetodoControle;
        this.fk_Cod_Praga = fk_Cod_Praga;
        this.fk_Cod_Talhao = fk_Cod_Talhao;
        this.Autor = Autor;
    }

    public AplicacaoModel() {
        this.Cod_Aplicacao = 0;
        this.Data = null;
        this.DataPlano = null;
        this.MetodoAplicado = null;
        this.popPragas = 0;
        this.numPlantas = 0;
        this.Autor = null;
        this.Estacao = null;
        this.Temperatura = null;
        this.Tempo = null;
        this.fk_Cod_MetodoControle = 0;
        this.fk_Cod_Praga = 0;
        this.fk_Cod_Talhao = 0;
    }


    public int getCod_Aplicacao() {
        return Cod_Aplicacao;
    }

    public void setCod_Aplicacao(int cod_Aplicacao) {
        this.Cod_Aplicacao = cod_Aplicacao;
    }

    public String getData() {
        return Data;
    }

    public void setData(String data) {
        this.Data = data;
    }

    public String getEstacao() {
        return Estacao;
    }

    public void setEstacao(String estacao) {
        this.Estacao = estacao;
    }

    public String getTemperatura() {
        return Temperatura;
    }

    public void setTemperatura(String temperatura) {
        this.Temperatura = temperatura;
    }

    public String getTempo() {
        return Tempo;
    }

    public void setTempo(String tempo) {
        this.Tempo = tempo;
    }

    public int getFk_Cod_MetodoControle() {
        return fk_Cod_MetodoControle;
    }

    public void setFk_Cod_MetodoControle(int fk_Cod_MetodoControle) {
        this.fk_Cod_MetodoControle = fk_Cod_MetodoControle;
    }

    public String getDataPlano() {
        return DataPlano;
    }

    public void setDataPlano(String dataPlano) {
        DataPlano = dataPlano;
    }

    public String getMetodoAplicado() {
        return MetodoAplicado;
    }

    public void setMetodoAplicado(String metodoAplicado) {
        MetodoAplicado = metodoAplicado;
    }

    public int getPopPragas() {
        return popPragas;
    }

    public void setPopPragas(int popPragas) {
        this.popPragas = popPragas;
    }

    public int getNumPlantas() {
        return numPlantas;
    }

    public void setNumPlantas(int numPlantas) {
        this.numPlantas = numPlantas;
    }

    public String getAutor() {
        return Autor;
    }

    public void setAutor(String autor) {
        Autor = autor;
    }

    public int getFk_Cod_Talhao() {
        return fk_Cod_Talhao;
    }

    public void setFk_Cod_Talhao(int fk_Cod_Talhao) {
        this.fk_Cod_Talhao = fk_Cod_Talhao;
    }

    public int getFk_Cod_Praga() {
        return fk_Cod_Praga;
    }

    public void setFk_Cod_Praga(int fk_Cod_Praga) {
        this.fk_Cod_Praga = fk_Cod_Praga;
    }
}
