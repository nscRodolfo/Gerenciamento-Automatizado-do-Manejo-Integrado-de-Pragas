package my.aplication.manejointeligentedepragas.model;

public class MetodoControleModel {
    private int Cod_MetodoControle;
    private String Nome;
    private String MateriaisNecessarios;
    private String ModoDePreparo;
    private String IntervaloAplicacao;
    private String EfeitoColateral;
    private String Observacoes;

    public MetodoControleModel(int cod_MetodoControle, String nome, String materiaisNecessarios, String modoDePreparo, String intervaloAplicacao, String efeitoColateral, String observacoes) {
        this.Cod_MetodoControle = cod_MetodoControle;
        this.Nome = nome;
        this.MateriaisNecessarios = materiaisNecessarios;
        this.ModoDePreparo = modoDePreparo;
        this.IntervaloAplicacao = intervaloAplicacao;
        this.EfeitoColateral = efeitoColateral;
        this.Observacoes = observacoes;
    }

    public int getCod_MetodoControle() {
        return Cod_MetodoControle;
    }

    public void setCod_MetodoControle(int cod_MetodoControle) {
        this.Cod_MetodoControle = cod_MetodoControle;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        this.Nome = nome;
    }

    public String getMateriaisNecessarios() {
        return MateriaisNecessarios;
    }

    public void setMateriaisNecessarios(String materiaisNecessarios) {
        this.MateriaisNecessarios = materiaisNecessarios;
    }

    public String getModoDePreparo() {
        return ModoDePreparo;
    }

    public void setModoDePreparo(String modoDePreparo) {
        this.ModoDePreparo = modoDePreparo;
    }

    public String getIntervaloAplicacao() {
        return IntervaloAplicacao;
    }

    public void setIntervaloAplicacao(String intervaloAplicacao) {
        this.IntervaloAplicacao = intervaloAplicacao;
    }

    public String getEfeitoColateral() {
        return EfeitoColateral;
    }

    public void setEfeitoColateral(String efeitoColateral) {
        this.EfeitoColateral = efeitoColateral;
    }

    public String getObservacoes() {
        return Observacoes;
    }

    public void setObservacoes(String observacoes) {
        this.Observacoes = observacoes;
    }
}
